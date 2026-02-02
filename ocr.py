import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from openvino.runtime import Core
import math
import os


class PaddleOCR:
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        char_dict_path: str,
        device: str = "CPU",
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.5,
        det_db_unclip_ratio: float = 1.6,
        det_limit_side_len: int = 960,
        det_min_box_size: int = 3,
        rec_batch_size: int = 6,
        rec_image_shape: str = "3,48,320",
        debug: bool = False,
        debug_dir: str = "./debug_crops",
        **kwargs
    ):
        self.device = device
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_limit_side_len = det_limit_side_len
        self.det_min_box_size = det_min_box_size
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = [int(x) for x in rec_image_shape.split(",")]
        self.debug = debug
        self.debug_dir = debug_dir
        
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        self.ie = Core()
        self._load_det_model(det_model_path)
        self._load_rec_model(rec_model_path)
        self.character = self._load_char_dict(char_dict_path)

    def _load_det_model(self, model_path: str):
        model = self.ie.read_model(model_path)
        self.det_model = self.ie.compile_model(model, self.device)
        self.det_input = self.det_model.input(0)
        self.det_output = self.det_model.output(0)

    def _load_rec_model(self, model_path: str):
        model = self.ie.read_model(model_path)
        self.rec_model = self.ie.compile_model(model, self.device)
        self.rec_input = self.rec_model.input(0)
        self.rec_output = self.rec_model.output(0)

    def _load_char_dict(self, dict_path: str) -> List[str]:
        character = ["blank"]
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                character.append(line.strip("\n"))
        character.append(" ")
        return character

    def _normalize_image(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        img = img / 255.0
        img -= 0.5
        img /= 0.5
        return img

    def _det_preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        src_h, src_w = img.shape[:2]
        
        ratio = 1.0
        if max(src_h, src_w) > self.det_limit_side_len:
            if src_h > src_w:
                ratio = float(self.det_limit_side_len) / src_h
            else:
                ratio = float(self.det_limit_side_len) / src_w
        
        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        resized = cv2.resize(img, (resize_w, resize_h))
        resized = self._normalize_image(resized)
        resized = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        shape_info = {
            'src_h': src_h,
            'src_w': src_w,
            'resize_h': resize_h,
            'resize_w': resize_w,
            'ratio_h': resize_h / float(src_h),
            'ratio_w': resize_w / float(src_w)
        }
        
        return resized, shape_info

    def _det_postprocess(self, pred: np.ndarray, shape_info: dict) -> List[np.ndarray]:
        pred = pred[0, 0]
        segmentation = pred > self.det_db_thresh
        
        mask = (segmentation * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue
            
            score = self._polygon_score(pred, contour)
            if score < self.det_db_box_thresh:
                continue
            
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box)
            
            box = self._unclip(box, self.det_db_unclip_ratio)
            if box is None:
                continue
            
            box = box.reshape(-1, 2)
            rect = cv2.minAreaRect(box)
            
            if min(rect[1]) < self.det_min_box_size:
                continue
            
            box = cv2.boxPoints(rect)
            box = self._order_points(box)
            
            box[:, 0] = np.clip(box[:, 0] / shape_info['ratio_w'], 0, shape_info['src_w'])
            box[:, 1] = np.clip(box[:, 1] / shape_info['ratio_h'], 0, shape_info['src_h'])
            
            boxes.append(box.astype(np.float32))
        
        return boxes

    def _polygon_score(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        h, w = bitmap.shape
        contour = contour.reshape(-1, 2)
        
        xmin = int(np.clip(np.floor(contour[:, 0].min()), 0, w - 1))
        xmax = int(np.clip(np.ceil(contour[:, 0].max()), 0, w - 1))
        ymin = int(np.clip(np.floor(contour[:, 1].min()), 0, h - 1))
        ymax = int(np.clip(np.ceil(contour[:, 1].max()), 0, h - 1))
        
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour_shifted = contour.copy()
        contour_shifted[:, 0] -= xmin
        contour_shifted[:, 1] -= ymin
        cv2.fillPoly(mask, [contour_shifted.astype(np.int32)], 1)
        
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box: np.ndarray, ratio: float) -> Optional[np.ndarray]:
        box = box.reshape(-1, 2)
        area = abs(cv2.contourArea(box))
        length = cv2.arcLength(box, True)
        
        if length < 1e-6:
            return None
        
        distance = area * ratio / length
        
        try:
            import pyclipper
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if not expanded:
                return None
            return np.array(expanded[0]).reshape(-1, 2).astype(np.float32)
        except ImportError:
            center = box.mean(axis=0)
            return (center + (box - center) * 1.5).astype(np.float32)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1).flatten()
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect(self, img: np.ndarray) -> List[np.ndarray]:
        input_tensor, shape_info = self._det_preprocess(img)
        result = self.det_model({self.det_input: input_tensor})
        pred = result[self.det_output]
        return self._det_postprocess(pred, shape_info)

    def _resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(imgH * max_wh_ratio)
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = min(resized_w, imgW)
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype(np.float32)
        resized_image = resized_image / 255.0
        resized_image -= 0.5
        resized_image /= 0.5
        
        padding_im = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        padding_im[:, 0:resized_w, :] = resized_image
        padding_im = padding_im.transpose(2, 0, 1)
        
        return padding_im

    def _rec_postprocess(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        
        for pred in preds:
            pred_idx = pred.argmax(axis=1)
            pred_prob = pred.max(axis=1)
            
            char_list = []
            conf_list = []
            last_idx = 0
            
            for idx, prob in zip(pred_idx, pred_prob):
                if idx != 0 and idx != last_idx:
                    if idx < len(self.character):
                        char_list.append(self.character[idx])
                        conf_list.append(prob)
                last_idx = idx
            
            text = ''.join(char_list)
            score = float(np.mean(conf_list)) if conf_list else 0.0
            results.append((text, score))
        
        return results

    def _recognize(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not img_list:
            return []
        
        img_num = len(img_list)
        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]
        indices = np.argsort(np.array(width_list))
        
        results = [None] * img_num
        
        for beg_img_no in range(0, img_num, self.rec_batch_size):
            end_img_no = min(img_num, beg_img_no + self.rec_batch_size)
            
            batch_indices = indices[beg_img_no:end_img_no]
            max_wh_ratio = max([width_list[i] for i in batch_indices])
            max_wh_ratio = max(max_wh_ratio, self.rec_image_shape[2] / float(self.rec_image_shape[1]))
            
            norm_img_batch = []
            for idx in batch_indices:
                norm_img = self._resize_norm_img(img_list[idx], max_wh_ratio)
                norm_img_batch.append(norm_img)
            
            norm_img_batch = np.array(norm_img_batch).astype(np.float32)
            
            preds = self.rec_model({self.rec_input: norm_img_batch})[self.rec_output]
            rec_results = self._rec_postprocess(preds)
            
            for i, idx in enumerate(batch_indices):
                results[idx] = rec_results[i]
        
        return results

    def _get_rotate_crop_image(self, img: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
        img_height, img_width = img.shape[0:2]
        points = points.astype(np.float32)
        
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        
        left = max(0, left - 3)
        top = max(0, top - 3)
        right = min(img_width, right + 3)
        bottom = min(img_height, bottom + 3)
        
        img_crop = img[top:bottom, left:right, :].copy()
        
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        
        img_crop_width = int(max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])
        ))
        img_crop_height = int(max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])
        ))
        
        if img_crop_width < 1 or img_crop_height < 1:
            return None
        
        pts_std = np.array([
            [0, 0],
            [img_crop_width, 0],
            [img_crop_width, img_crop_height],
            [0, img_crop_height]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img_crop,
            M,
            (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        
        return dst_img

    def _sorted_boxes(self, dt_boxes: List[np.ndarray]) -> List[np.ndarray]:
        if len(dt_boxes) == 0:
            return []
        
        num_boxes = len(dt_boxes)
        sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
        _boxes = list(sorted_boxes)
        
        for i in range(num_boxes - 1):
            for j in range(i, -1, -1):
                if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                   (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                    tmp = _boxes[j]
                    _boxes[j] = _boxes[j + 1]
                    _boxes[j + 1] = tmp
                else:
                    break
        
        return _boxes

    def predict(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Could not load image: {img}")
        else:
            image = img.copy()
        
        ori_im = image.copy()
        
        dt_boxes = self._detect(image)
        if not dt_boxes:
            return []
        
        dt_boxes = self._sorted_boxes(dt_boxes)
        
        if self.debug:
            debug_img = ori_im.copy()
            for i, box in enumerate(dt_boxes):
                box_int = box.astype(np.int32)
                cv2.polylines(debug_img, [box_int], True, (0, 255, 0), 2)
                cv2.putText(debug_img, str(i), (box_int[0][0], box_int[0][1] - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(self.debug_dir, "detection_boxes.jpg"), debug_img)
        
        img_crop_list = []
        valid_boxes = []
        for i, box in enumerate(dt_boxes):
            tmp_box = box.copy()
            img_crop = self._get_rotate_crop_image(ori_im, tmp_box)
            if img_crop is not None:
                img_crop_list.append(img_crop)
                valid_boxes.append(box)
                if self.debug:
                    cv2.imwrite(os.path.join(self.debug_dir, f"crop_{i}.jpg"), img_crop)
        
        if not img_crop_list:
            return []
        
        rec_results = self._recognize(img_crop_list)
        
        results = []
        for i, (text, score) in enumerate(rec_results):
            results.append({
                'text_region': valid_boxes[i].tolist(),
                'text': text,
                'confidence': score
            })
            if self.debug:
                print(f"Crop {i}: '{text}' (conf: {score:.4f})")
        
        return results

    def __call__(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        return self.predict(img)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python paddleocr_wrapper.py <det_model.xml> <rec_model.xml> <char_dict.txt> <image_path> [--debug]")
        sys.exit(0)
    
    debug_mode = "--debug" in sys.argv
    
    ocr = PaddleOCR(
        det_model_path=sys.argv[1],
        rec_model_path=sys.argv[2],
        char_dict_path=sys.argv[3],
        debug=debug_mode
    )
    
    result = ocr.predict(sys.argv[4])
    
    texts = []
    for r in result:
        texts.append(r['text'].replace("'", ""))
    print(texts)