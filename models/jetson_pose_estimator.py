import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import logging
from adaptive_object_detection.detectors.jetson_detector import JetsonDetector
from .base_pose_estimator import BasePoseEstimator
from tools.convert_results_format import prepare_detection_results
from tools.bbox import box_to_center_scale, center_scale_to_box
from tools.pose_nms import pose_nms
from tools.transformations import get_affine_transform, get_max_pred
import numpy as np
import cv2
import time


class TRTPoseEstimator(BasePoseEstimator):
    def __init__(self,
                 detector_thresh,
                 detector_input_size=(300, 300),
                 pose_input_size=(256, 192),
                 heatmap_size=(64, 48),
                 batch_size=8,
                 pose_model_path=None
                 ):
        super().__init__(detector_thresh)
        self.detector_height, self.detector_width = detector_input_size
        self.pose_input_size = pose_input_size
        self.heatmap_size = heatmap_size
        self.batch_size = batch_size
        self.pose_model_path = pose_model_path
        self.h_input = None
        self.d_input = None
        self.h_ouput = None
        self.d_output = None
        self.stream = None
        self.raw_frame = None

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.model = None
        self.detector = None

    def load_model(self, detector_path, detector_label_map):
        self.detector = JetsonDetector(width=self.detector_width, height=self.detector_height,
                                       thresh=self.detector_thresh)
        self.detector.load_model(detector_path, detector_label_map)
        self._init_cuda_stuff()

    def _batch_execute(self, context):
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        context.execute(batch_size=self.batch_size, bindings=[int(self.d_input), int(self.d_output)])
        cuda.memcpy_dtoh_async(self.h_ouput, self.d_output, self.stream)
        result_raw = self.h_ouput.reshape((self.batch_size, 64, 48, 17))  # TODO: it only works for fastpost
        return result_raw

    def inference(self, preprocessed_image): 
        start_time_A = time.perf_counter()
        start_time = time.perf_counter()
        raw_detections = self.detector.inference(preprocessed_image)
        finish_time = time.perf_counter()
        #print("|--> Detection time = ", (finish_time - start_time) * 1000)
        
        start_time = time.perf_counter()
        detections = prepare_detection_results(raw_detections, self.detector_width, self.detector_height)
        #print("Number of objects ==========================", np.shape(detections))

        '''detections = np.array([
                np.array([0., 126.39, 154.81, 142.69, 212.77, 0.90, 0.99, 0]),
                np.array([0., 204.98, 38.77, 216.33, 82.06, 0.77, 0.99, 0]
),
                np.array([0., 178.29, 109.85, 188.02, 153.39, 0.77, 0.99, 0]),
                np.array([0., 156.37, 49.41, 169.97, 88.59, 0.73, 0.99, 0]),
                np.array([0., 180.93, 159.36, 193.91, 207.87, 0.71, 0.99, 0]),
                np.array([0., 206.64, 142.76, 217.35, 182.08, 0.63, 0.99, 0]),
                np.array([0., 236.62, 80.77, 252.72, 144.48, 0.62, 0.99, 0]),
                np.array([0., 12, 3.96, 264.93, 138.02, 0.60, 0.99, 0]),
                np.array([0., 191.37, 11.17, 202.12, 150.43, 0.5, 0.99, 0]),
                np.array([0., 185.33, 54.53, 191.88, 76.61, 0.45, 0.99, 0])
                
                ])'''
        finish_time = time.perf_counter()
        #print("|--> Detection perepare time = ", (finish_time - start_time) * 1000)
        
        resized_pose_img = cv2.resize(self.raw_frame, (self.detector_width, self.detector_height))
        rgb_resized_img = cv2.cvtColor(resized_pose_img, cv2.COLOR_BGR2RGB)
        start_time = time.perf_counter()
        inps, cropped_boxes, boxes, scores, ids = self.transform_detections(rgb_resized_img, detections)
        finish_time = time.perf_counter()
        #print("|--> transform detections time = ", (finish_time - start_time) * 1000)
        
        if inps.shape[0] == 0:
            return (None, None, None, None, None)
        start_time = time.perf_counter()
        num_detected_objects = np.shape(inps)[0]
        batch_inps = np.zeros([self.batch_size, self.pose_input_size[0], self.pose_input_size[1], 3])
        result = np.zeros([num_detected_objects, 64, 48, 17])
        if num_detected_objects < self.batch_size:
            batch_inps[0:num_detected_objects, :] = inps
            self._load_images_to_buffer(batch_inps)
            with self.model.create_execution_context() as context:
                # Transfer input data to the GPU.
                result_raw = self._batch_execute(context)
                result = result_raw[0:num_detected_objects, :]

        else:
            remainder = num_detected_objects
            start_idx = 0
            while remainder > 0:
                endidx = min(self.batch_size, remainder)
                batch_inps[0:endidx, :] = inps[start_idx: start_idx + endidx, :]
                self._load_images_to_buffer(batch_inps)
                with self.model.create_execution_context() as context:
                    result_raw = self._batch_execute(context)
                    result[start_idx: start_idx + endidx, :] = result_raw[0:endidx, :]
                remainder -= self.batch_size
                start_idx += self.batch_size
        finish_time = time.perf_counter()
       # print("|--> Batch creation time",(finish_time - start_time) * 1000)
        finish_time_A = time.perf_counter()
        #print("|--> Inference interior time = ", (finish_time_A - start_time_A) * 1000)
        return (result, cropped_boxes, boxes, scores, ids)

    def preprocess(self, raw_image):
        self.raw_frame = raw_image
        return self.detector.preprocess(raw_image)

    def post_process(self, hm, cropped_boxes, boxes, scores, ids):
        if hm is None:
            return

        assert hm.ndim == 4
        pose_coords = []
        pose_scores = []
        for i in range(hm.shape[0]):
            start_time = time.perf_counter()
            hm_size = self.heatmap_size
            eval_joints = list(range(17))
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = self.heatmap_to_coord(hm[i, :, :, eval_joints], bbox, hm_shape=hm_size,
                                                           norm_type=None)
            finish_time = time.perf_counter()
         #   print("|----> postprocess heatmap to coord time (per object) = ", (finish_time - start_time) * 1000)

            pose_coords.append(pose_coord)
            pose_scores.append(pose_score)

        preds_img = np.array(pose_coords)
        preds_scores = np.array(pose_scores)
        start_time = time.perf_counter()
        #boxes, scores, ids, preds_img, preds_scores, pick_ids = \
        #    pose_nms(boxes, scores, ids, preds_img, preds_scores, 0)
        finish_time = time.perf_counter()
        #print("|----> postprocess pose nms time = ", (finish_time - start_time) * 1000)

        _result = []
        for k in range(len(scores)):
            start_time = time.perf_counter()
            if np.ndim(preds_scores[k] == 2):
                preds_scores[k] = preds_scores[k][:, 0].reshape([17, 1])
                _result.append(
                    {
                        'keypoints': preds_img[k],
                        'kp_score': preds_scores[k],
                        'proposal_score': np.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                        'idx': ids[k],
                        'bbox': [boxes[k][0], boxes[k][1], boxes[k][2], boxes[k][3]]
                    }
                )
            finish_time = time.perf_counter()
         #   print("|----> postprocess keypoints and result prepration time = ", (finish_time - start_time) * 1000)
        return _result
    

    def transform_detections(self, image, dets):
        # image = image.transpose(2,1,0)
        start_time = time.perf_counter()
        input_size = self.pose_input_size
        if isinstance(dets, int):
            return 0, 0
        dets = dets[dets[:, 0] == 0]
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]
        ids = np.zeros(scores.shape)
        inps = np.zeros([boxes.shape[0], int(input_size[0]), int(input_size[1]), 3])
        cropped_boxes = np.zeros([boxes.shape[0], 4])
        finish_time = time.perf_counter()
        #print("|----> transform detections variables filling time = ", (finish_time - start_time) * 1000)
        
        start_time = time.perf_counter()
        # ===================== func1 ======================
        scale_mult = 1.25
        aspect_ratio = input_size[1] / input_size[0] 
        centers = np.zeros((boxes.shape[0], 2), dtype=np.float32)
        w = boxes[:,2] - boxes[:,0]
        h = boxes[:,3] - boxes[:,1]
        centers[:,0] = boxes[:, 0] + w * 0.5
        centers[:,1] = boxes[:, 1] + h * 0.5
        aspect_ratio = input_size[1] / input_size[0]
        idx = np.where(np.array((w > aspect_ratio * h), dtype=int) > 0)
        h[idx] = w[idx] / aspect_ratio
        idx = np.where(np.array((w < aspect_ratio * h), dtype=int) > 0 )
        w[idx] = h[idx] * aspect_ratio
        scales = np.zeros((boxes.shape[0], 2), dtype=np.float32)
        scales[:,0] = w
        scales[:,1] = h
        idx = np.where(centers[:,0] != -1)
        scales[idx,:] = scales[idx,:] * scale_mult
        #bboxes = np.zeros((boxes.shape[0],4), dtype=np.float32)
        xmin = np.array(centers[:,0] - scales[:,0] * 0.5)
        ymin = np.array(centers[:,1] - scales[:,1] * 0.5)

        cropped_boxes = np.array([xmin, ymin,np.array(xmin + scales[:,0]),
                np.array(ymin + scales[:,1])])
        cropped_boxes = np.transpose(cropped_boxes)
        
        # ====================== func2 ========================
        rot = 0
        scales_tmp = scales
        src_w = scales_tmp[:,0]
        dst_w = input_size[1]
        dst_h = input_size[0]

        rot_rad = np.pi * rot / 180
        src_results = np.zeros(scales.shape, dtype=np.float32)
        src_points = np.zeros(scales.shape, dtype=np.float32)
        src_points[:,1] = src_w * -0.5
        
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_results = np.zeros(scales.shape, dtype=np.float32)
        src_results[:,0] = src_points[:,0] * cs - src_points[:,1] * sn
        src_results[:,1] = src_points[:,0] * sn + src_points[:,1] * cs
        
        src_dir = src_results
        
        dst_dir = np.zeros(scales.shape, dtype=np.float32)
        dst_dir[:,1] = dst_w * -0.5

        shift = np.zeros(scales.shape, dtype=np.float32)
        src = np.zeros((scales.shape[0],3,2), dtype=np.float32)
        dst = np.zeros((scales.shape[0],3,2), dtype=np.float32)
        src[:,0,:] = centers + scales_tmp * shift
        src[:,1,:] = centers + src_dir + scales_tmp * shift
        dst[:,0,:] = [dst_w * 0.5, dst_h * 0.5]
        dst[:,1,:] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        #------------
        direct_1 = src[:,0,:] - src[:,1,:]
        a = np.array([-direct_1[:,1], direct_1[:,0]])
        a = np.transpose(a)
        src[:,2,:] = src[:,1,:] +  a
        
        direct_2 = dst[:,0,:] - dst[:,1,:]
        b = np.array([-direct_2[:,1], direct_2[:,0]])
        b = np.transpose(b)
        dst[:,2,:] = dst[:,1,:] + b
        #------------
        finish_time = time.perf_counter()
        
        #print("****-> transform get affine prep/wrap/center time = ",
                #(finish_time - start_time) * 1000)
        image = image / 255.0
        image[..., 0] = image[..., 0] - 0.406
        image[..., 1] = image[..., 1] - 0.457
        image[..., 2] = image[..., 2] - 0.480
        for i, itm in enumerate(dst):
            start_time = time.perf_counter()
            trans = cv2.getAffineTransform(np.float32(src[i,:,:]), np.float32(itm))
            print("vectorized trans:" ,trans)
            inps[i] = cv2.warpAffine(image, trans, (int(input_size[1]), int(input_size[0])), flags=cv2.INTER_LINEAR)
            finish_time = time.perf_counter()

        print("*-------------------------------*")
        v_c = cropped_boxes.copy()
        v_inps = inps.copy()
        #return inps, cropped_boxes, boxes, scores, ids
        #exit()
        
        for i, box in enumerate(boxes):
            start_time = time.perf_counter()
            inps[i], cropped_box = self.transform_single_detection(image, box, input_size)
            finish_time = time.perf_counter()
            #print("|----> transform single detection time = ", (finish_time - start_time) * 1000)
            cropped_boxes[i] = np.float32(cropped_box)
        exit()
        print(np.all(inps == v_inps))
        print(np.all(cropped_boxes == v_c))
        print("------------------------------------")
        print(type(v_c))
        print(type(cropped_boxes))
        print(type(inps))
        print(type(v_inps))
        print(np.where(v_inps != inps))
        if not (np.all(inps == v_inps)) or not (np.all(cropped_boxes == v_c)):
            print("vectors are not equal!")
            exit()
        exit()
        # inps = im_to_tensor(inps)
        return inps, cropped_boxes, boxes, scores, ids
        

    @staticmethod
    def transform_single_detection(image, bbox, input_size):
        aspect_ratio = input_size[1] / input_size[0]
        xmin, ymin, xmax, ymax = bbox
        start_time = time.perf_counter()
        center, scale = box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, aspect_ratio)
        finish_time = time.perf_counter()
        #print("|------> transform single box to center time = ", (finish_time - start_time) * 1000)
        
        scale = scale * 1.0

        input_size = input_size
        
        inp_h, inp_w = input_size

        start_time = time.perf_counter()
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
        print("original: ", trans)
        finish_time = time.perf_counter()
        #print("|------> transform single get affine transform time = ", (finish_time - start_time) * 1000)
        
        inp_h, inp_w = input_size
        start_time = time.perf_counter()
        img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        finish_time = time.perf_counter()
        #print("|------> transform single wrapAffine time = ", (finish_time - start_time) * 1000)

        start_time = time.perf_counter()
        bbox = center_scale_to_box(center, scale)
        finish_time = time.perf_counter()
        #print("|------> transform single center scale to box time = ", (finish_time - start_time) * 1000)
        
        img = img / 255.0
        img[..., 0] = img[..., 0] - 0.406
        img[..., 1] = img[..., 1] - 0.457
        img[..., 2] = img[..., 2] - 0.480
        # img = im_to_tensor(img)
        return img, bbox

    def heatmap_to_coord(self, hms, bbox, hms_flip=None, **kwargs):
        if hms_flip is not None:
            hms = (hms + hms_flip) / 2
        if not isinstance(hms, np.ndarray):
            hms = hms.cpu().data.numpy()
        coords, maxvals = get_max_pred(hms)

        hm_h = hms.shape[1]
        hm_w = hms.shape[2]

        # post-processing
        for p in range(coords.shape[0]):
            hm = hms[p]
            px = int(round(float(coords[p][0])))
            py = int(round(float(coords[p][1])))
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                diff = np.array((hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px]))
                coords[p] += np.sign(diff) * .25

        preds = np.zeros_like(coords)

        # transform bbox to scale
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        center = np.array([xmin + w * 0.5, ymin + h * 0.5])
        scale = np.array([w, h])
        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = self.transform_preds(coords[i], center, scale,
                                            [hm_w, hm_h])

        return preds, maxvals

    def transform_preds(self, coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = get_affine_transform(center, scale, 0, output_size, inv=1)
        target_coords[0:2] = self.affine_transform(coords[0:2], trans)
        return target_coords

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def _load_images_to_buffer(self, img):
        preprocessed = np.asarray(img).ravel()
        np.copyto(self.h_input, preprocessed)

    def _load_engine(self):
        model_path =self.pose_model_path
        if not os.path.isfile(model_path):
            logging.info(
                'model does not exist under: {}'.format(str(model_path)))
        else:
            with open(model_path, 'rb') as f:
                engine_data = f.read()
            engine = self.trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def _init_cuda_stuff(self):
        self.model = self._load_engine()
        self.h_input, self.d_input, self.h_ouput, self.d_output, self.stream = self._allocate_buffers(self.model,
                                                                                                      self.batch_size,
                                                                                                      trt.float32)

    @staticmethod
    def _allocate_buffers(engine, batch_size, data_type):
        """
        This is the function to allocate buffers for input and output in the device
        Args:
           engine : The path to the TensorRT engine.
           batch_size : The batch size for execution time.
           data_type: The type of the data for input and output, for example trt.float32.

        Output:
           h_input_1: Input in the host.
           d_input_1: Input in the device.
           h_output_1: Output in the host.
           d_output_1: Output in the device.
           stream: CUDA stream.

        """
        # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
        h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)),
                                          dtype=trt.nptype(data_type))
        h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)),
                                         dtype=trt.nptype(data_type))
        # Allocate device memory for inputs and outputs.
        d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

        d_output = cuda.mem_alloc(h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()
        return h_input_1, d_input_1, h_output, d_output, stream
