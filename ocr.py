“””
Drop-in replacement for PaddleOCR using OpenVINO.
Provides the exact same interface as your original code:

```
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)
result = ocr.predict("test_img.jpg")
```

Usage:
from paddleocr_wrapper import PaddleOCR
ocr = PaddleOCR(
use_doc_orientation_classify=False,
use_doc_unwarping=False,
use_textline_orientation=False
)
result = ocr.predict(“test_img.jpg”)
“””

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from openvino.runtime import Core
import math

class PaddleOCR:
“””
Drop-in replacement for PaddleOCR using OpenVINO inference.

```
This class provides the same API as paddleocr.PaddleOCR
but uses OpenVINO IR models instead.
"""

def __init__(
    self,
    # Original PaddleOCR parameters (for compatibility)
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_textline_orientation: bool = False,
    # OpenVINO specific parameters
    model_dir: str = "./models",
    det_model_name: str = "det_model",
    rec_model_name: str = "rec_model",
    cls_model_name: str = "cls_model",
    char_dict_path: str = None,
    device: str = "CPU",
    # Detection parameters - matched to PaddleOCR v5 defaults
    det_db_thresh: float = 0.3,
    det_db_box_thresh: float = 0.5,  # Changed from 0.6 to match PaddleOCR default
    det_db_unclip_ratio: float = 1.6,  # Changed from 1.5 to match PaddleOCR default
    det_limit_side_len: int = 960,
    det_min_box_size: int = 3,  # Minimum box size
    # Recognition parameters
    rec_batch_size: int = 6,
    rec_image_shape: str = "3,48,320",
    # Classification parameters
    use_angle_cls: bool = None,
    cls_batch_size: int = 6,
    cls_thresh: float = 0.9,
    **kwargs  # Accept any other parameters for compatibility
):
    """
    Initialize PaddleOCR with OpenVINO backend.
    
    Args:
        use_doc_orientation_classify: Ignored (for API compatibility)
        use_doc_unwarping: Ignored (for API compatibility)
        use_textline_orientation: Controls angle classification
        model_dir: Directory containing OpenVINO IR models
        det_model_name: Name of detection model (without extension)
        rec_model_name: Name of recognition model (without extension)
        cls_model_name: Name of classification model (without extension)
        char_dict_path: Path to character dictionary
        device: OpenVINO device (CPU, GPU, etc.)
    """
    self.model_dir = Path(model_dir)
    self.device = device
    
    # Map original parameters to OpenVINO parameters
    self.use_angle_cls = use_textline_orientation if use_angle_cls is None else use_angle_cls
    
    # Detection parameters
    self.det_db_thresh = det_db_thresh
    self.det_db_box_thresh = det_db_box_thresh
    self.det_db_unclip_ratio = det_db_unclip_ratio
    self.det_limit_side_len = det_limit_side_len
    self.det_min_box_size = det_min_box_size
    
    # Recognition parameters
    self.rec_batch_size = rec_batch_size
    self.rec_image_shape = [int(x) for x in rec_image_shape.split(",")]
    
    # Classification parameters
    self.cls_batch_size = cls_batch_size
    self.cls_thresh = cls_thresh
    self.cls_image_shape = [3, 48, 192]
    
    # Character dictionary path
    if char_dict_path is None:
        char_dict_path = self.model_dir / "ppocr_keys_v1.txt"
    self.char_dict_path = Path(char_dict_path)
    
    # Initialize OpenVINO
    self.ie = Core()
    
    # Load models
    self._load_models(det_model_name, rec_model_name, cls_model_name)
    
    # Load character dictionary
    self.character = self._load_char_dict()

def _load_models(self, det_name: str, rec_name: str, cls_name: str):
    """Load all required models"""
    # Detection model
    det_path = self.model_dir / f"{det_name}.xml"
    if det_path.exists():
        print(f"Loading detection model: {det_path}")
        model = self.ie.read_model(str(det_path))
        self.det_model = self.ie.compile_model(model, self.device)
        self.det_input = self.det_model.input(0)
        self.det_output = self.det_model.output(0)
    else:
        raise FileNotFoundError(f"Detection model not found: {det_path}")
    
    # Recognition model
    rec_path = self.model_dir / f"{rec_name}.xml"
    if rec_path.exists():
        print(f"Loading recognition model: {rec_path}")
        model = self.ie.read_model(str(rec_path))
        self.rec_model = self.ie.compile_model(model, self.device)
        self.rec_input = self.rec_model.input(0)
        self.rec_output = self.rec_model.output(0)
    else:
        raise FileNotFoundError(f"Recognition model not found: {rec_path}")
    
    # Classification model (optional)
    cls_path = self.model_dir / f"{cls_name}.xml"
    if cls_path.exists() and self.use_angle_cls:
        print(f"Loading classification model: {cls_path}")
        model = self.ie.read_model(str(cls_path))
        self.cls_model = self.ie.compile_model(model, self.device)
        self.cls_input = self.cls_model.input(0)
        self.cls_output = self.cls_model.output(0)
    else:
        self.cls_model = None
        if self.use_angle_cls:
            print(f"Warning: Classification model not found: {cls_path}")
            self.use_angle_cls = False

def _load_char_dict(self) -> List[str]:
    """Load character dictionary for CTC decoding"""
    character = []
    
    if self.char_dict_path.exists():
        with open(self.char_dict_path, "r", encoding="utf-8") as f:
            for line in f:
                character.append(line.strip("\n"))
    else:
        print(f"Warning: Character dictionary not found: {self.char_dict_path}")
        # Default ASCII character set
        character = list(
            "0123456789abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        )
    
    # Add blank token for CTC
    return ["blank"] + character

# ==================== Detection ====================

def _det_preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
    """Preprocess image for detection - matches PaddleOCR resize logic"""
    src_h, src_w = img.shape[:2]
    
    # Calculate resize ratio while preserving aspect ratio
    ratio = 1.0
    if max(src_h, src_w) > self.det_limit_side_len:
        if src_h > src_w:
            ratio = self.det_limit_side_len / src_h
        else:
            ratio = self.det_limit_side_len / src_w
    
    resize_h = int(src_h * ratio)
    resize_w = int(src_w * ratio)
    
    # Round to multiple of 32
    resize_h = max(int(round(resize_h / 32) * 32), 32)
    resize_w = max(int(round(resize_w / 32) * 32), 32)
    
    # Calculate actual ratios after rounding
    ratio_h = resize_h / src_h
    ratio_w = resize_w / src_w
    
    resized = cv2.resize(img, (resize_w, resize_h))
    
    # Normalize (ImageNet normalization)
    resized = resized.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    resized = (resized / 255.0 - mean) / std
    
    # HWC -> NCHW
    resized = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    
    return resized, (ratio_h, ratio_w, resize_h, resize_w)

def _det_postprocess(
    self, pred: np.ndarray, src_h: int, src_w: int, shape_info: Tuple[float, float, int, int]
) -> List[np.ndarray]:
    """Extract bounding boxes from detection output - matches PaddleOCR DBPostProcess"""
    ratio_h, ratio_w, resize_h, resize_w = shape_info
    pred = pred[0, 0]  # Remove batch and channel dims
    segmentation = pred > self.det_db_thresh
    
    # Find contours on the binary mask
    mask = (segmentation * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    scores = []
    
    for contour in contours:
        # Filter by contour area (very small regions)
        if cv2.contourArea(contour) < 4:
            continue
        
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box)
        
        # Calculate box score from the probability map
        score = self._box_score(pred, contour)
        if score < self.det_db_box_thresh:
            continue
        
        # Unclip the box to expand it
        box = self._unclip(box, self.det_db_unclip_ratio)
        if box is None:
            continue
        
        # Get the minimum area rect of the expanded box
        box = box.reshape(-1, 2)
        rect = cv2.minAreaRect(box)
        box = cv2.boxPoints(rect)
        box = np.array(box)
        
        # Filter by minimum size
        if min(rect[1]) < self.det_min_box_size:
            continue
        
        # Order the points consistently
        box = self._order_points(box)
        
        # Scale coordinates back to original image size using separate ratios
        box[:, 0] = box[:, 0] / ratio_w
        box[:, 1] = box[:, 1] / ratio_h
        
        # Clip to image boundaries
        box[:, 0] = np.clip(box[:, 0], 0, src_w - 1)
        box[:, 1] = np.clip(box[:, 1], 0, src_h - 1)
        
        boxes.append(box.astype(np.float32))
        scores.append(score)
    
    return boxes

def _box_score(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
    """Calculate mean score inside contour - matches PaddleOCR implementation"""
    h, w = bitmap.shape[:2]
    contour = contour.reshape(-1, 2)
    
    xmin = np.clip(np.floor(contour[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(contour[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(contour[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(contour[:, 1].max()).astype(np.int32), 0, h - 1)
    
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    
    # Shift contour to local coordinates
    contour_shifted = contour.copy()
    contour_shifted[:, 0] = contour_shifted[:, 0] - xmin
    contour_shifted[:, 1] = contour_shifted[:, 1] - ymin
    
    cv2.fillPoly(mask, [contour_shifted.astype(np.int32)], 1)
    
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

def _unclip(self, box: np.ndarray, ratio: float) -> Optional[np.ndarray]:
    """Expand box using unclip ratio - matches PaddleOCR implementation"""
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
        
        # Get the largest expanded polygon
        expanded = sorted(expanded, key=lambda x: cv2.contourArea(np.array(x)), reverse=True)
        return np.array(expanded[0]).reshape(-1, 2)
        
    except ImportError:
        # Fallback: simple centroid-based expansion
        center = box.mean(axis=0)
        expanded = center + (box - center) * (1 + distance / max(np.linalg.norm(box - center, axis=1).mean(), 1e-6))
        return expanded.astype(np.float32)

def _order_points(self, pts: np.ndarray) -> np.ndarray:
    """Order points: TL, TR, BR, BL"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def _detect(self, img: np.ndarray) -> List[np.ndarray]:
    """Run detection inference"""
    src_h, src_w = img.shape[:2]
    input_tensor, shape_info = self._det_preprocess(img)
    result = self.det_model({self.det_input: input_tensor})
    pred = result[self.det_output]
    return self._det_postprocess(pred, src_h, src_w, shape_info)

# ==================== Classification ====================

def _cls_preprocess(self, img: np.ndarray) -> np.ndarray:
    """Preprocess for classification"""
    imgC, imgH, imgW = self.cls_image_shape
    h, w = img.shape[:2]
    
    ratio = w / float(h)
    resized_w = min(imgW, int(math.ceil(imgH * ratio)))
    
    resized = cv2.resize(img, (resized_w, imgH))
    resized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    padded = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padded[:, :resized_w, :] = resized
    
    return padded.transpose(2, 0, 1)

def _classify(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
    """Classify text direction"""
    if not self.use_angle_cls or self.cls_model is None:
        return [("0", 1.0)] * len(img_list)
    
    results = []
    for i in range(0, len(img_list), self.cls_batch_size):
        batch = img_list[i:i + self.cls_batch_size]
        inputs = np.stack([self._cls_preprocess(img) for img in batch])
        pred = self.cls_model({self.cls_input: inputs})[self.cls_output]
        
        for j in range(len(batch)):
            idx = pred[j].argmax()
            score = float(pred[j].max())
            results.append((str(idx), score))
    
    return results

# ==================== Recognition ====================

def _rec_preprocess(self, img: np.ndarray) -> np.ndarray:
    """Preprocess for recognition"""
    imgC, imgH, imgW = self.rec_image_shape
    h, w = img.shape[:2]
    
    ratio = w / float(h)
    resized_w = min(imgW, int(math.ceil(imgH * ratio)))
    
    resized = cv2.resize(img, (resized_w, imgH))
    resized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    
    padded = np.zeros((imgH, imgW, imgC), dtype=np.float32)
    padded[:, :resized_w, :] = resized
    
    return padded.transpose(2, 0, 1)

def _rec_postprocess(self, pred: np.ndarray) -> List[Tuple[str, float]]:
    """CTC decode recognition output"""
    results = []
    
    for b in range(pred.shape[0]):
        pred_idx = pred[b].argmax(axis=1)
        
        chars, confs = [], []
        prev = 0
        for t in range(len(pred_idx)):
            idx = pred_idx[t]
            if idx != 0 and idx != prev and idx < len(self.character):
                chars.append(self.character[idx])
                confs.append(float(pred[b, t, idx]))
            prev = idx
        
        text = "".join(chars)
        conf = float(np.mean(confs)) if confs else 0.0
        results.append((text, conf))
    
    return results

def _recognize(self, img_list: List[np.ndarray]) -> List[Tuple[str, float]]:
    """Run recognition inference"""
    if not img_list:
        return []
    
    results = []
    for i in range(0, len(img_list), self.rec_batch_size):
        batch = img_list[i:i + self.rec_batch_size]
        inputs = np.stack([self._rec_preprocess(img) for img in batch])
        pred = self.rec_model({self.rec_input: inputs})[self.rec_output]
        results.extend(self._rec_postprocess(pred))
    
    return results

# ==================== Crop ====================

def _crop_image(self, img: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """Crop and straighten text region"""
    box = box.astype(np.float32)
    
    width = int(max(
        np.linalg.norm(box[0] - box[1]),
        np.linalg.norm(box[2] - box[3])
    ))
    height = int(max(
        np.linalg.norm(box[0] - box[3]),
        np.linalg.norm(box[1] - box[2])
    ))
    
    if width < 1 or height < 1:
        return None
    
    dst_pts = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(box, dst_pts)
    cropped = cv2.warpPerspective(
        img, M, (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC
    )
    
    if cropped.shape[0] > cropped.shape[1] * 1.5:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
    
    return cropped

def _sort_boxes(self, boxes: List[np.ndarray]) -> List[np.ndarray]:
    """Sort boxes top-to-bottom, left-to-right"""
    return sorted(boxes, key=lambda b: (b[0][1], b[0][0]))

# ==================== Main API ====================

def predict(
    self, 
    img: Union[str, np.ndarray],
    cls: bool = None,
    det: bool = True,
    rec: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run OCR on image.
    
    Args:
        img: Image path or numpy array (RGB or BGR)
        cls: Override angle classification setting
        det: Run detection
        rec: Run recognition
        
    Returns:
        List of dictionaries with keys:
        - text_region: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        - text: recognized text string
        - confidence: recognition confidence (0-1)
    """
    # Load image
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
    
    # Detection
    boxes = self._detect(image)
    if not boxes:
        return []
    
    boxes = self._sort_boxes(boxes)
    
    # Crop
    crops, valid_boxes = [], []
    for box in boxes:
        crop = self._crop_image(image, box)
        if crop is not None and crop.size > 0:
            crops.append(crop)
            valid_boxes.append(box)
    
    if not crops:
        return []
    
    # Classification
    use_cls = cls if cls is not None else self.use_angle_cls
    if use_cls:
        cls_results = self._classify(crops)
        for i, (label, score) in enumerate(cls_results):
            if label == "1" and score > self.cls_thresh:
                crops[i] = cv2.rotate(crops[i], cv2.ROTATE_180)
    
    # Recognition
    rec_results = self._recognize(crops)
    
    # Format output (matches PaddleOCR format)
    results = []
    for box, (text, conf) in zip(valid_boxes, rec_results):
        if text.strip():
            results.append({
                'text_region': box.tolist(),
                'text': text,
                'confidence': conf
            })
    
    return results

def ocr(
    self, 
    img: Union[str, np.ndarray],
    cls: bool = None,
    det: bool = True,
    rec: bool = True,
) -> List[List[Any]]:
    """
    Alternative API matching PaddleOCR.ocr() output format.
    
    Returns:
        [
            [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)],
            ...
        ]
    """
    results = self.predict(img, cls=cls, det=det, rec=rec)
    
    return [
        [r['text_region'], (r['text'], r['confidence'])]
        for r in results
    ]

def __call__(
    self, 
    img: Union[str, np.ndarray],
    **kwargs
) -> List[Dict[str, Any]]:
    """Alias for predict()"""
    return self.predict(img, **kwargs)
```

# ==================== Example ====================

if **name** == “**main**”:
import sys

```
print("PaddleOCR OpenVINO Drop-in Replacement")
print("=" * 50)

if len(sys.argv) < 2:
    print("\nUsage: python paddleocr_wrapper.py <image_path>")
    print("\nExpected files in ./models/:")
    print("  - det_model.xml, det_model.bin")
    print("  - rec_model.xml, rec_model.bin")
    print("  - ppocr_keys_v1.txt")
    sys.exit(0)

# Same API as original code
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

result = ocr.predict(sys.argv[1])

print(f"\nFound {len(result)} text regions:\n")
for i, r in enumerate(result):
    print(f"[{i+1}] {r['text']} ({r['confidence']:.4f})")
    print(f"    Region: {r['text_region']}\n")
```