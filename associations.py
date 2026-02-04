import supervision as sv
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class Association:
    """Represents an association between a product/slot and its SKU label(s)."""
    detection_idx: int
    detection_class: str
    sku_label_idxs: List[int]
    bbox: np.ndarray


def associate_sku_labels(
    detections: sv.Detections,
    img_height: int,
    img_width: int,
    class_names: Optional[List[str]] = None,
    shelf_gap_ratio: float = 0.08,
    alignment_tolerance_ratio: float = 0.03,
) -> List[Association]:
    if class_names is None:
        class_names = ['object', 'empty_slot', 'sku_label']

    labels = [class_names[cid] for cid in detections.class_id]

    items = []
    sku_labels = []

    for i, (bbox, cls) in enumerate(zip(detections.xyxy, labels)):
        if cls in ('object', 'empty_slot'):
            items.append((i, bbox, cls))
        elif cls == 'sku_label':
            sku_labels.append((i, bbox))

    if not items or not sku_labels:
        return []

    shelf_gap_tol = img_height * shelf_gap_ratio
    align_tol = img_width * alignment_tolerance_ratio

    def cy(bbox): return (bbox[1] + bbox[3]) / 2.0
    def cx(bbox): return (bbox[0] + bbox[2]) / 2.0
    def left(bbox): return bbox[0]
    def right(bbox): return bbox[2]
    def bottom(bbox): return bbox[3]

    sorted_labels = sorted(sku_labels, key=lambda s: cy(s[1]))
    shelf_rows = []

    for lbl in sorted_labels:
        placed = False
        for row in shelf_rows:
            row_cy = np.mean([cy(b) for _, b in row])
            if abs(cy(lbl[1]) - row_cy) < shelf_gap_tol:
                row.append(lbl)
                placed = True
                break
        if not placed:
            shelf_rows.append([lbl])

    for row in shelf_rows:
        row.sort(key=lambda s: left(s[1]))

    associations = []

    for item_idx, item_bbox, item_cls in items:
        item_bottom_y = bottom(item_bbox)
        item_center_x = cx(item_bbox)
        item_left_x = left(item_bbox)
        item_right_x = right(item_bbox)

        best_row = None
        best_dist = float('inf')

        for row in shelf_rows:
            row_cy = np.mean([cy(b) for _, b in row])
            dist = row_cy - item_bottom_y
            if dist < -shelf_gap_tol:
                continue
            if abs(dist) < best_dist:
                best_dist = abs(dist)
                best_row = row

        if best_row is None:
            for row in shelf_rows:
                row_cy = np.mean([cy(b) for _, b in row])
                dist = abs(row_cy - item_bottom_y)
                if dist < best_dist:
                    best_dist = dist
                    best_row = row

        if best_row is None:
            continue

        if item_cls == 'empty_slot':
            spanning = [
                lbl_idx for lbl_idx, lbl_bbox in best_row
                if right(lbl_bbox) > item_left_x + align_tol
                and left(lbl_bbox) < item_right_x - align_tol
            ]
            if spanning:
                associations.append(Association(
                    detection_idx=item_idx,
                    detection_class=item_cls,
                    sku_label_idxs=spanning,
                    bbox=item_bbox,
                ))
                continue

        best_label_idx = None
        best_label_dist = float('inf')

        for lbl_idx, lbl_bbox in best_row:
            lbl_left = left(lbl_bbox)
            if lbl_left <= item_center_x + align_tol:
                dist = item_center_x - lbl_left
                if dist < best_label_dist:
                    best_label_dist = dist
                    best_label_idx = lbl_idx

        if best_label_idx is None:
            for lbl_idx, lbl_bbox in best_row:
                dist = abs(left(lbl_bbox) - item_center_x)
                if dist < best_label_dist:
                    best_label_dist = dist
                    best_label_idx = lbl_idx

        if best_label_idx is not None:
            associations.append(Association(
                detection_idx=item_idx,
                detection_class=item_cls,
                sku_label_idxs=[best_label_idx],
                bbox=item_bbox,
            ))

    return associations


# --- colors per class ---
CLASS_COLORS = {
    'object':     (0, 255, 0),    # green
    'empty_slot': (0, 0, 255),    # red
    'sku_label':  (255, 165, 0),  # orange
}

# association line colors
ASSOC_COLORS = {
    'object':     (0, 220, 0),
    'empty_slot': (50, 50, 255),
}


def draw_associations(
    image: np.ndarray,
    detections: sv.Detections,
    associations: List[Association],
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    dot_radius: int = 5,
) -> np.ndarray:
    """
    Draw bounding boxes for all detections and association lines
    from each object/empty_slot center to its associated sku_label center(s).
    """
    if class_names is None:
        class_names = ['object', 'empty_slot', 'sku_label']

    vis = image.copy()

    # --- draw all bounding boxes first ---
    for i, bbox in enumerate(detections.xyxy):
        cls_name = class_names[detections.class_id[i]]
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thickness)

        label = cls_name.replace('_', ' ')
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # --- draw association lines ---
    for assoc in associations:
        item_bbox = detections.xyxy[assoc.detection_idx]
        item_cx = int((item_bbox[0] + item_bbox[2]) / 2)
        item_cy = int((item_bbox[1] + item_bbox[3]) / 2)
        line_color = ASSOC_COLORS.get(assoc.detection_class, (255, 255, 255))

        # dot at item center
        cv2.circle(vis, (item_cx, item_cy), dot_radius, line_color, -1)

        for lbl_idx in assoc.sku_label_idxs:
            lbl_bbox = detections.xyxy[lbl_idx]
            lbl_cx = int((lbl_bbox[0] + lbl_bbox[2]) / 2)
            lbl_cy = int((lbl_bbox[1] + lbl_bbox[3]) / 2)

            # line from item center to label center
            cv2.line(vis, (item_cx, item_cy), (lbl_cx, lbl_cy),
                     line_color, line_thickness, cv2.LINE_AA)

            # dot at label center
            cv2.circle(vis, (lbl_cx, lbl_cy), dot_radius, line_color, -1)

    return vis