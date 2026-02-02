import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from openvino.runtime import Core
import math


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
        use_space_char: bool = True,
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
        self.use_space_char = use_space_char
        
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
        character = []
        with open(dict_path, "r", encoding="utf-8") as f:
            for line in f:
                character.append(line.strip("\n"))
        if self.use_space_char and " " not in character:
            character.append(" ")
        return ["blank"] + character

    def _det_preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
        src_h, src_w = img.shape[:2]
        ratio = 1.0
        if max(src_h, src_w) > self.det_limit_side_len:
            if src_h > src_w:
                ratio = self.det_limit_side_len / src_h
            else:
                ratio = self.det_limit_side_len / src_w
        
        resize_h = int(src_h * ratio)
        resize_w = int(src_w * ratio)
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        ratio_h = resize_h / src_h
        ratio_w = resize_w / src_w
        
        resized = cv2.resize(img, (resize_w, resize_h))
        resized = resized.astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        return resized, (ratio_h, ratio_w, resize_h, resize_w)

    def _det_postprocess(
        self, pred: np.ndarray, src_h: int, src_w: int, shape_info: Tuple[float, float, int, int]
    ) -> List[np.ndarray]:
        ratio_h, ratio_w, resize_h, resize_w = shape_info
        pred = pred[0, 0]
        segmentation = pred > self.det_db_thresh
        
        mask = (segmentation * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue
            
            score = self._box_score(pred, contour)
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
            box = cv2.boxPoints(rect)
            box = np.array(box)
            
            if min(rect[1]) < self.det_min_box_size:
                continue
            
            box = self._order_points(box)
            box[:, 0] = box[:, 0] / ratio_w
            box[:, 1] = box[:, 1] / ratio_h
            box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)
            
            boxes.append(box.astype(np.float32))
        
        return boxes

    def _box_score(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        h, w = bitmap.shape[:2]
        contour = contour.reshape(-1, 2)
        
        xmin = np.clip(np.floor(contour[:, 0].min()).astype(np.int32), 0, w - 1)
        xmax = np.clip(np.ceil(contour[:, 0].max()).astype(np.int32), 0, w - 1)
        ymin = np.clip(np.floor(contour[:, 1].min()).astype(np.int32), 0, h - 1)
        ymax = np.clip(np.ceil(contour[:, 1].max()).astype(np.int32), 0, h - 1)
        
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour_shifted = contour.copy()
        contour_shifted[:, 0] = contour_shifted[:, 0] - xmin
        contour_shifted[:, 1] = contour_shifted[:, 1] - ymin
        cv2.fillPoly(mask, [contour_shifted.astype(np.int32)], 1)
        
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def _unclip(self, box: np.ndarray, ratio: float) -> Optional[np.ndarray]:
        box = box.reshape(-1, 2)
        area = cv2.contourArea(box)
        length = cv2.arcLength(box, True)
        
        if length == 0:
            return None
        
        distance = area * ratio / length
        
        try:
            import pyclipper
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            if not expanded:
                return None
            expanded = sorted(expanded, key=lambda x: cv2.contourArea(np.array(x)), reverse=True)
            return np.array(expanded[0]).reshape(-1, 2)
        except ImportError:
            center = box.mean(axis=0)
            expanded = center + (box - center) * (1 + distance / max(np.linalg.norm(box - center, axis=1).mean(), 1e-6))
            return expanded.astype(np.float32)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _detect(self, img: np.ndarray) -> List[np.ndarray]:
        src_h, src_w = img.shape[:2]
        input_tensor, shape_info = self._det_preprocess(img)
        result = self.det_model({self.det_input: input_tensor})
        pred = result[self.det_output]
        return self._det_postprocess(pred, src_h, src_w, shape_info)

    def _rec_preprocess(self, img: np.ndarray, max_wh_ratio: float = None) -> np.ndarray:
        imgC, imgH, imgW = self.rec_image_shape
        if max_wh_ratio is not None:
            imgW = int(imgH * max_wh_ratio)
            imgW = max(min(imgW, 1280), 32)
        
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = int(math.ceil(imgH * ratio))
        resized_w = min(resized_w, imgW)
        
        resized = cv2.resize(img, (resized_w, imgH))
        resized = resized.astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5
        
        padded = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        padded[:, :resized_w, :] = resized
        
        return padded.transpose(2, 0, 1)

    def _rec_postprocess(self, pred: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for b in range(pred.shape[0]):
            pred_prob = pred[b]
            pred_idx = pred_prob.argmax(axis=1)
            pred_scores = pred_prob.max(axis=1)
            
            chars, confs = [], []
            prev = 0
            for t in range(len(pred_idx)):
                idx = int(pred_idx[t])
                if idx != 0 and idx != prev:
                    if idx < len(self.character):
                        chars.append(self.character[idx])
                        confs.append(float(pred_scores[t]))
                prev = idx
            
            text = "".join(chars)
            conf = float(np.mean(confs)) if confs else 0.0
            results.append((text, conf))
        return results

    def _recognize(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not img_list:
            return []
        
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        
        indices = np.argsort(np.array(width_list))
        
        results = [None] * len(img_list)
        batch_num = math.ceil(len(img_list) / self.rec_batch_size)
        
        for bn in range(batch_num):
            start = bn * self.rec_batch_size
            end = min((bn + 1) * self.rec_batch_size, len(img_list))
            
            batch_indices = indices[start:end]
            max_wh_ratio = max([width_list[i] for i in batch_indices])
            
            batch_inputs = []
            for idx in batch_indices:
                img = img_list[idx]
                preprocessed = self._rec_preprocess(img, max_wh_ratio)
                batch_inputs.append(preprocessed)
            
            batch_array = np.stack(batch_inputs)
            pred = self.rec_model({self.rec_input: batch_array})[self.rec_output]
            batch_results = self._rec_postprocess(pred)
            
            for i, idx in enumerate(batch_indices):
                results[idx] = batch_results[i]
        
        return results

    def _crop_image(self, img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        box = box.astype(np.float32)
        
        img_crop_width = int(max(
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[2] - box[3])
        ))
        img_crop_height = int(max(
            np.linalg.norm(box[0] - box[3]),
            np.linalg.norm(box[1] - box[2])
        ))
        
        if img_crop_width < 1 or img_crop_height < 1:
            return None
        
        dst_pts = np.array([
            [0, 0],
            [img_crop_width - 1, 0],
            [img_crop_width - 1, img_crop_height - 1],
            [0, img_crop_height - 1]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(box, dst_pts)
        cropped = cv2.warpPerspective(
            img, M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC
        )
        
        if img_crop_height > img_crop_width * 1.5:
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        return cropped

    def _sort_boxes(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        if len(boxes) == 0:
            return boxes
        
        boxes_info = []
        for i, box in enumerate(boxes):
            y_top = min(box[0][1], box[1][1])
            y_bottom = max(box[2][1], box[3][1])
            x_left = min(box[0][0], box[3][0])
            boxes_info.append({
                'idx': i,
                'box': box,
                'y_top': y_top,
                'y_bottom': y_bottom,
                'y_center': (y_top + y_bottom) / 2,
                'x_left': x_left,
                'height': y_bottom - y_top
            })
        
        avg_height = np.mean([b['height'] for b in boxes_info])
        
        boxes_info.sort(key=lambda x: x['y_center'])
        
        lines = []
        current_line = [boxes_info[0]]
        
        for b in boxes_info[1:]:
            last_in_line = current_line[-1]
            if abs(b['y_center'] - last_in_line['y_center']) < avg_height * 0.6:
                current_line.append(b)
            else:
                lines.append(current_line)
                current_line = [b]
        lines.append(current_line)
        
        sorted_boxes = []
        for line in lines:
            line.sort(key=lambda x: x['x_left'])
            for b in line:
                sorted_boxes.append(b['box'])
        
        return sorted_boxes

    def predict(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Could not load image: {img}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = img.copy()
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        
        boxes = self._detect(image)
        if not boxes:
            return []
        
        boxes = self._sort_boxes(boxes)
        
        crops, valid_boxes = [], []
        for box in boxes:
            crop = self._crop_image(image, box)
            if crop is not None and crop.size > 0:
                crops.append(crop)
                valid_boxes.append(box)
        
        if not crops:
            return []
        
        rec_results = self._recognize(crops)
        
        results = []
        for box, (text, conf) in zip(valid_boxes, rec_results):
            results.append({
                'text_region': box.tolist(),
                'text': text,
                'confidence': conf
            })
        
        return results

    def __call__(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        return self.predict(img)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python paddleocr_wrapper.py <det_model.xml> <rec_model.xml> <char_dict.txt> <image_path>")
        sys.exit(0)
    
    ocr = PaddleOCR(
        det_model_path=sys.argv[1],
        rec_model_path=sys.argv[2],
        char_dict_path=sys.argv[3]
    )
    
    result = ocr.predict(sys.argv[4])
    
    texts = []
    for r in result:
        texts.append(r['text'].replace("'", ""))
    print(texts)