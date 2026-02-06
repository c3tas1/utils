import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from openvino.runtime import Core
import math
import os


class PaddleOCR:
    def __init__(
        self,
        det_model_path: str = None,
        rec_model_path: str = None,
        char_dict_path: str = None,
        device: str = "CPU",
        det_db_thresh: float = 0.15,
        det_db_box_thresh: float = 0.2,
        det_db_unclip_ratio: float = 2.5,
        det_limit_side_len: int = 960,
        det_limit_type: str = "max",
        det_min_box_size: int = 3,
        det_scales: list = None,
        rec_batch_size: int = 6,
        rec_image_shape: str = "3,48,320",
        preprocess: bool = True,
        upscale_small: bool = True,
        min_image_size: int = 1200,
        use_craft: bool = False,
        craft_model_path: str = None,
    ):
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.det_limit_side_len = det_limit_side_len
        self.det_limit_type = det_limit_type
        self.det_min_box_size = det_min_box_size
        self.det_scales = det_scales if det_scales else [80, 160, 320, 640, 960]
        self.rec_batch_size = rec_batch_size
        self.rec_image_shape = [int(x) for x in rec_image_shape.split(",")]
        self.preprocess = preprocess
        self.upscale_small = upscale_small
        self.min_image_size = min_image_size
        self.use_craft = use_craft

        self.ie = Core()

        if det_model_path and not use_craft:
            det_model = self.ie.read_model(det_model_path)
            self.det_compiled = self.ie.compile_model(det_model, device)
            self.det_input = self.det_compiled.input(0)
            self.det_output = self.det_compiled.output(0)
        else:
            self.det_compiled = None

        if use_craft:
            self._init_craft(craft_model_path, device)

        if rec_model_path:
            rec_model = self.ie.read_model(rec_model_path)
            self.rec_compiled = self.ie.compile_model(rec_model, device)
            self.rec_input = self.rec_compiled.input(0)
            self.rec_output = self.rec_compiled.output(0)

        self.character = ["blank"]
        if char_dict_path:
            with open(char_dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.character.append(line.strip("\n"))
            self.character.append(" ")

    def _init_craft(self, model_path: str, device: str):
        if model_path:
            craft_model = self.ie.read_model(model_path)
            self.craft_compiled = self.ie.compile_model(craft_model, device)
            self.craft_input = self.craft_compiled.input(0)
            self.craft_output_y = self.craft_compiled.output(0)
        else:
            self.craft_compiled = None
            try:
                import craft_text_detector
                self.craft_detector = craft_text_detector
            except ImportError:
                print("CRAFT not available. Install with: pip install craft-text-detector")
                self.craft_detector = None

    def _enhance_image(self, img: np.ndarray) -> np.ndarray:
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
        kernel = np.array([[0, -1, 0],
                          [-1,  5, -1],
                          [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        return enhanced

    def _upscale_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if max(h, w) < self.min_image_size:
            scale = self.min_image_size / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

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
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
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

    def _detect_craft(self, img: np.ndarray) -> List[np.ndarray]:
        if hasattr(self, 'craft_detector') and self.craft_detector:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                cv2.imwrite(f.name, img)
                prediction_result = self.craft_detector.detect_text(
                    f.name,
                    text_threshold=0.7,
                    link_threshold=0.4,
                    low_text=0.4,
                    cuda=False,
                    poly=False
                )
                os.unlink(f.name)
            
            boxes = []
            for box in prediction_result['boxes']:
                box = np.array(box).reshape(4, 2).astype(np.float32)
                boxes.append(box)
            return boxes
        return []

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
        if self.use_craft:
            return self._detect_craft(img)

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

    def _crop_text(self, img: np.ndarray, pts: np.ndarray, padding: int = 8) -> Optional[np.ndarray]:
        h, w = img.shape[:2]
        pts = pts.astype(np.float32).reshape(4, 2)

        left, right = max(0, int(pts[:, 0].min()) - padding), min(w, int(pts[:, 0].max()) + padding)
        top, bottom = max(0, int(pts[:, 1].min()) - padding), min(h, int(pts[:, 1].max()) + padding)

        crop = img[top:bottom, left:right, :].copy()
        pts_shifted = pts - np.array([left, top], dtype=np.float32)

        cw = int(max(np.linalg.norm(pts_shifted[0] - pts_shifted[1]), np.linalg.norm(pts_shifted[2] - pts_shifted[3])))
        ch = int(max(np.linalg.norm(pts_shifted[0] - pts_shifted[3]), np.linalg.norm(pts_shifted[1] - pts_shifted[2])))
        if cw < 1 or ch < 1:
            return None

        cw += padding * 2
        ch += padding * 2

        src = pts_shifted.reshape(4, 2).astype(np.float32)
        dst = np.array([[padding, padding], [cw - padding, padding],
                       [cw - padding, ch - padding], [padding, ch - padding]], dtype=np.float32)
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

    def predict(self, img: Union[str, np.ndarray], save_debug: str = None) -> List[Dict[str, Any]]:
        if isinstance(img, str):
            image = cv2.imread(img)
            if image is None:
                raise ValueError(f"Could not load image: {img}")
        else:
            image = img.copy()

        original = image.copy()

        if self.upscale_small:
            image = self._upscale_image(image)

        if self.preprocess:
            image = self._enhance_image(image)

        processed = image.copy()

        boxes = self._sort_boxes(self._detect(image))
        if not boxes:
            if save_debug:
                self._save_debug_image(original, processed, [], [], save_debug)
            return []

        crops, valid = [], []
        for box in boxes:
            crop = self._crop_text(image, box.copy())
            if crop is not None:
                crops.append(crop)
                valid.append(box)

        if not crops:
            if save_debug:
                self._save_debug_image(original, processed, boxes, [], save_debug)
            return []

        rec_results = self._recognize(crops)

        results = [
            {'text_region': valid[i].tolist(), 'text': text, 'confidence': conf, 'crop': crops[i]}
            for i, (text, conf) in enumerate(rec_results)
        ]

        if save_debug:
            self._save_debug_image(original, processed, valid, results, save_debug)

        for r in results:
            del r['crop']

        return results

    def _save_debug_image(self, original: np.ndarray, processed: np.ndarray,
                          boxes: List[np.ndarray], results: List[Dict], save_path: str):
        oh, ow = original.shape[:2]
        ph, pw = processed.shape[:2]

        scale_w = ow / pw
        scale_h = oh / ph

        left_img = original.copy()
        right_img = cv2.resize(processed, (ow, oh))

        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0)
        ]

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            box_orig = box.copy()
            box_orig[:, 0] = box[:, 0] * scale_w
            box_orig[:, 1] = box[:, 1] * scale_h
            box_scaled = box_orig.astype(np.int32)

            cv2.polylines(left_img, [box_scaled], True, color, 2)
            cv2.polylines(right_img, [box_scaled], True, color, 2)

            if i < len(results):
                text = results[i]['text']
                conf = results[i]['confidence']
                label = f"{i}: {text} ({conf:.2f})"

                x, y = int(box_scaled[0][0]), int(box_scaled[0][1]) - 5
                cv2.putText(left_img, label, (x, max(y, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        combined = np.hstack([left_img, right_img])

        if results:
            crop_height = 60
            num_crops = len(results)
            crops_per_row = max(1, combined.shape[1] // 200)
            num_rows = (num_crops + crops_per_row - 1) // crops_per_row
            crop_panel_height = num_rows * (crop_height + 40)
            crop_panel = np.ones((crop_panel_height, combined.shape[1], 3), dtype=np.uint8) * 255

            x_offset = 10
            y_offset = 10
            max_crop_width = 180

            for i, res in enumerate(results):
                crop = res['crop']
                ch, cw = crop.shape[:2]

                ratio = min(crop_height / ch, max_crop_width / cw)
                new_w, new_h = int(cw * ratio), int(ch * ratio)
                resized_crop = cv2.resize(crop, (new_w, new_h))

                if x_offset + new_w + 10 > combined.shape[1]:
                    x_offset = 10
                    y_offset += crop_height + 40

                if y_offset + new_h < crop_panel_height:
                    crop_panel[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_crop

                    text = res['text'] if len(res['text']) <= 20 else res['text'][:17] + "..."
                    label = f"{i}: {text}"
                    cv2.putText(crop_panel, label, (x_offset, y_offset + new_h + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                x_offset += new_w + 20

            combined = np.vstack([combined, crop_panel])

        cv2.imwrite(save_path, combined)

    def __call__(self, img: Union[str, np.ndarray], save_debug: str = None) -> List[Dict[str, Any]]:
        return self.predict(img, save_debug)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print("Usage: python paddleocr_openvino.py <det_model.xml> <rec_model.xml> <char_dict.txt> <image_path> [output_debug.jpg]")
        print("\nFor CRAFT detection:")
        print("  pip install craft-text-detector")
        print("  Then use: use_craft=True in constructor")
        sys.exit(0)

    ocr = PaddleOCR(
        det_model_path=sys.argv[1],
        rec_model_path=sys.argv[2],
        char_dict_path=sys.argv[3],
    )

    debug_path = sys.argv[5] if len(sys.argv) > 5 else "debug_output.jpg"
    result = ocr.predict(sys.argv[4], save_debug=debug_path)

    print(f"\nDetected {len(result)} text regions:")
    print("-" * 50)
    for i, r in enumerate(result):
        print(f"{i}: '{r['text']}' (conf: {r['confidence']:.2f})")
    print("-" * 50)
    print(f"Debug image saved to: {debug_path}")