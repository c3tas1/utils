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
        det_limit_type: str = "max",
        det_min_box_size: int = 3,
        rec_batch_size: int = 6,
        rec_image_shape: str = "3,48,320",
        det_scales: list = None,
    ):
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_min_box_size = det_min_box_size
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = [int(x) for x in rec_image_shape.split(",")]
        self.det_scales = det_scales

        self.ie = Core()

        det_model = self.ie.read_model(det_model_path)
        self.det_compiled = self.ie.compile_model(det_model, device)
        self.det_input = self.det_compiled.input(0)
        self.det_output = self.det_compiled.output(0)

        rec_model = self.ie.read_model(rec_model_path)
        self.rec_compiled = self.ie.compile_model(rec_model, device)
        self.rec_input = self.rec_compiled.input(0)
        self.rec_output = self.rec_compiled.output(0)

        self.character = ["blank"]
        with open(char_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                self.character.append(line.strip("\n"))
        self.character.append(" ")

    def _det_preprocess(self, img: np.ndarray, limit_side_len: int = None, limit_type: str = None) -> Tuple[np.ndarray, dict]:
        src_h, src_w = img.shape[:2]
        limit = limit_side_len or self.det_limit_side_len
        ltype = limit_type or self.det_limit_type
        ratio = 1.0

        if ltype == "min":
            if min(src_h, src_w) < limit:
                ratio = limit / min(src_h, src_w)
        else:
            if max(src_h, src_w) > limit:
                ratio = limit / max(src_h, src_w)

        resize_h = max(int(round(int(src_h * ratio) / 32) * 32), 32)
        resize_w = max(int(round(int(src_w * ratio) / 32) * 32), 32)

        resized = cv2.resize(img, (resize_w, resize_h)).astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5
        resized = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        return resized, {
            'src_h': src_h, 'src_w': src_w,
            'ratio_h': resize_h / src_h, 'ratio_w': resize_w / src_w
        }

    def _det_postprocess(self, pred: np.ndarray, shape: dict) -> List[np.ndarray]:
        pred = pred[0, 0]
        mask = ((pred > self.det_db_thresh) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue

            pts = contour.reshape(-1, 2)
            xmin, xmax = int(pts[:, 0].min()), int(pts[:, 0].max())
            ymin, ymax = int(pts[:, 1].min()), int(pts[:, 1].max())
            if xmax <= xmin or ymax <= ymin:
                continue

            roi_mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
            cv2.fillPoly(roi_mask, [pts - [xmin, ymin]], 1)
            score = cv2.mean(pred[ymin:ymax + 1, xmin:xmax + 1], roi_mask)[0]
            if score < self.det_db_box_thresh:
                continue

            box = cv2.boxPoints(cv2.minAreaRect(contour))
            box = self._unclip(box, self.det_db_unclip_ratio)
            if box is None:
                continue

            rect = cv2.minAreaRect(box.reshape(-1, 2))
            if min(rect[1]) < self.det_min_box_size:
                continue

            box = self._order_points(cv2.boxPoints(rect))
            box[:, 0] = np.clip(box[:, 0] / shape['ratio_w'], 0, shape['src_w'])
            box[:, 1] = np.clip(box[:, 1] / shape['ratio_h'], 0, shape['src_h'])
            boxes.append(box.astype(np.float32))

        return boxes

    def _unclip(self, box: np.ndarray, ratio: float) -> Optional[np.ndarray]:
        box = box.reshape(-1, 2)
        area, length = abs(cv2.contourArea(box)), cv2.arcLength(box, True)
        if length < 1e-6:
            return None
        distance = area * ratio / length
        try:
            import pyclipper
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(distance)
            return np.array(expanded[0]).astype(np.float32) if expanded else None
        except ImportError:
            center = box.mean(axis=0)
            return (center + (box - center) * 1.5).astype(np.float32)

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s, d = pts.sum(axis=1), np.diff(pts, axis=1).flatten()
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(d)], pts[np.argmax(d)]
        return rect

    def _run_det(self, img: np.ndarray, limit_side_len: int = None, limit_type: str = None) -> List[np.ndarray]:
        tensor, shape = self._det_preprocess(img, limit_side_len, limit_type)
        pred = self.det_compiled({self.det_input: tensor})[self.det_output]
        return self._det_postprocess(pred, shape)

    def _boxes_overlap(self, a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> bool:
        ax1, ay1 = a[:, 0].min(), a[:, 1].min()
        ax2, ay2 = a[:, 0].max(), a[:, 1].max()
        bx1, by1 = b[:, 0].min(), b[:, 1].min()
        bx2, by2 = b[:, 0].max(), b[:, 1].max()

        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)

        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        min_area = min(area_a, area_b)

        return (inter / min_area) > threshold if min_area > 0 else False

    def _merge_boxes(self, boxes_a: List[np.ndarray], boxes_b: List[np.ndarray]) -> List[np.ndarray]:
        if not boxes_a:
            return boxes_b
        if not boxes_b:
            return boxes_a

        merged = list(boxes_a)
        for b in boxes_b:
            is_dup = False
            for a in merged:
                if self._boxes_overlap(a, b, 0.5):
                    is_dup = True
                    break
            if not is_dup:
                merged.append(b)
        return merged

    def _detect(self, img: np.ndarray) -> List[np.ndarray]:
        boxes = self._run_det(img)

        if self.det_scales:
            for scale in self.det_scales:
                if scale == self.det_limit_side_len:
                    continue
                extra = self._run_det(img, limit_side_len=scale, limit_type="max")
                boxes = self._merge_boxes(boxes, extra)

        return boxes

    def _rec_preprocess(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        imgC, imgH, imgW = self.rec_image_shape
        imgW = int(imgH * max_wh_ratio)
        h, w = img.shape[:2]
        resized_w = min(int(math.ceil(imgH * w / h)), imgW)

        resized = cv2.resize(img, (resized_w, imgH)).astype(np.float32)
        resized = (resized / 255.0 - 0.5) / 0.5

        padded = np.zeros((imgH, imgW, imgC), dtype=np.float32)
        padded[:, :resized_w, :] = resized
        return padded.transpose(2, 0, 1)

    def _rec_postprocess(self, preds: np.ndarray) -> List[Tuple[str, float]]:
        results = []
        for pred in preds:
            indices, probs = pred.argmax(axis=1), pred.max(axis=1)
            chars, confs, last = [], [], 0
            for idx, prob in zip(indices, probs):
                if idx != 0 and idx != last and idx < len(self.character):
                    chars.append(self.character[idx])
                    confs.append(prob)
                last = idx
            results.append((''.join(chars), float(np.mean(confs)) if confs else 0.0))
        return results

    def _recognize(self, crops: List[np.ndarray]) -> List[Tuple[str, float]]:
        if not crops:
            return []

        n = len(crops)
        ratios = [img.shape[1] / img.shape[0] for img in crops]
        indices = np.argsort(ratios)
        results = [None] * n

        for i in range(0, n, self.rec_batch_size):
            batch_idx = indices[i:min(i + self.rec_batch_size, n)]
            max_ratio = max(ratios[j] for j in batch_idx)
            max_ratio = max(max_ratio, self.rec_image_shape[2] / self.rec_image_shape[1])

            batch = np.stack([self._rec_preprocess(crops[j], max_ratio) for j in batch_idx]).astype(np.float32)
            preds = self.rec_compiled({self.rec_input: batch})[self.rec_output]

            for k, idx in enumerate(batch_idx):
                results[idx] = self._rec_postprocess(preds[k:k + 1])[0]

        return results

    def _crop_text(self, img: np.ndarray, pts: np.ndarray) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        pts = pts.astype(np.float32).reshape(4, 2)

        left, right = max(0, int(pts[:, 0].min()) - 3), min(w, int(pts[:, 0].max()) + 3)
        top, bottom = max(0, int(pts[:, 1].min()) - 3), min(h, int(pts[:, 1].max()) + 3)

        crop = img[top:bottom, left:right, :].copy()
        pts_shifted = pts - np.array([left, top], dtype=np.float32)

        cw = int(max(np.linalg.norm(pts_shifted[0] - pts_shifted[1]), np.linalg.norm(pts_shifted[2] - pts_shifted[3])))
        ch = int(max(np.linalg.norm(pts_shifted[0] - pts_shifted[3]), np.linalg.norm(pts_shifted[1] - pts_shifted[2])))
        if cw < 1 or ch < 1:
            return None

        src = pts_shifted.reshape(4, 2).astype(np.float32)
        dst = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(crop, M, (cw, ch), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)

        return np.rot90(out) if out.shape[0] / out.shape[1] >= 1.5 else out

    def _sort_boxes(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda x: (x[0][1], x[0][0]))
        for i in range(len(boxes) - 1):
            for j in range(i, -1, -1):
                if abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10 and boxes[j + 1][0][0] < boxes[j][0][0]:
                    boxes[j], boxes[j + 1] = boxes[j + 1], boxes[j]
                else:
                    break
        return boxes

    def predict(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Could not load image: {img}")
        else:
            image = img.copy()

        boxes = self._sort_boxes(self._detect(image))
        if not boxes:
            return []

        crops, valid = [], []
        for box in boxes:
            crop = self._crop_text(image, box.copy())
            if crop is not None:
                crops.append(crop)
                valid.append(box)

        if not crops:
            return []

        rec_results = self._recognize(crops)
        return [
            {'text_region': valid[i].tolist(), 'text': text, 'confidence': conf}
            for i, (text, conf) in enumerate(rec_results)
        ]

    def __call__(self, img: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        return self.predict(img)