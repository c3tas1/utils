import supervision as sv
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2


# class name mapping
CLASS_NAMES = ['Object', 'EmptySlot', 'SkuLabel', 'OfferLabel', 'Reflection']

# classes that act as labels (SkuLabel + Reflection)
LABEL_CLASSES = {'SkuLabel', 'Reflection'}
# classes that are association targets
TARGET_CLASSES = {'Object', 'EmptySlot'}


def associate_sku_labels(
    detections: sv.Detections,
    img_height: int,
    img_width: int,
    class_names: Optional[List[str]] = None,
    shelf_gap_ratio: float = 0.08,
    alignment_tolerance_ratio: float = 0.03,
) -> List[dict]:
    """
    Associate Objects and EmptySlots with their SKU labels based on
    NIST shelf-label positioning guidelines.

    Returns:
        List of dicts with keys:
            label_idx:   index into original detections for the label
            label_xyxy:  [x1, y1, x2, y2] bbox of the label
            target_xyxy: [x1, y1, x2, y2] bbox of the associated object/slot
            target_type: class name of the target ('Object' or 'EmptySlot')
    """
    if class_names is None:
        class_names = CLASS_NAMES

    labels_list = [class_names[cid] for cid in detections.class_id]

    # separate targets and label-like detections
    targets = []     # (det_idx, bbox, class_name)
    sku_labels = []  # (det_idx, bbox)

    for i, (bbox, cls) in enumerate(zip(detections.xyxy, labels_list)):
        if cls in TARGET_CLASSES:
            targets.append((i, bbox, cls))
        elif cls in LABEL_CLASSES:
            sku_labels.append((i, bbox))

    if not targets or not sku_labels:
        return []

    shelf_gap_tol = img_height * shelf_gap_ratio
    align_tol = img_width * alignment_tolerance_ratio

    def cy(bbox): return (bbox[1] + bbox[3]) / 2.0
    def cx(bbox): return (bbox[0] + bbox[2]) / 2.0
    def left(bbox): return bbox[0]
    def right(bbox): return bbox[2]
    def bottom(bbox): return bbox[3]

    # cluster label detections into shelf rows by vertical center
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

    # build associations
    results = []

    for item_idx, item_bbox, item_cls in targets:
        item_bottom_y = bottom(item_bbox)
        item_center_x = cx(item_bbox)
        item_left_x = left(item_bbox)
        item_right_x = right(item_bbox)

        # find the label row just below this target
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

        # EmptySlot spanning multiple labels
        if item_cls == 'EmptySlot':
            spanning = [
                (lbl_idx, lbl_bbox) for lbl_idx, lbl_bbox in best_row
                if right(lbl_bbox) > item_left_x + align_tol
                and left(lbl_bbox) < item_right_x - align_tol
            ]
            if spanning:
                for lbl_idx, lbl_bbox in spanning:
                    results.append({
                        'label_idx': lbl_idx,
                        'label_xyxy': lbl_bbox.tolist() if hasattr(lbl_bbox, 'tolist') else list(lbl_bbox),
                        'target_xyxy': item_bbox.tolist() if hasattr(item_bbox, 'tolist') else list(item_bbox),
                        'target_type': item_cls,
                    })
                continue

        # standard: single label association
        best_label_idx = None
        best_label_bbox = None
        best_label_dist = float('inf')

        for lbl_idx, lbl_bbox in best_row:
            lbl_left = left(lbl_bbox)
            if lbl_left <= item_center_x + align_tol:
                dist = item_center_x - lbl_left
                if dist < best_label_dist:
                    best_label_dist = dist
                    best_label_idx = lbl_idx
                    best_label_bbox = lbl_bbox

        # fallback: closest label overall
        if best_label_idx is None:
            for lbl_idx, lbl_bbox in best_row:
                dist = abs(left(lbl_bbox) - item_center_x)
                if dist < best_label_dist:
                    best_label_dist = dist
                    best_label_idx = lbl_idx
                    best_label_bbox = lbl_bbox

        if best_label_idx is not None:
            results.append({
                'label_idx': best_label_idx,
                'label_xyxy': best_label_bbox.tolist() if hasattr(best_label_bbox, 'tolist') else list(best_label_bbox),
                'target_xyxy': item_bbox.tolist() if hasattr(item_bbox, 'tolist') else list(item_bbox),
                'target_type': item_cls,
            })

    return results


# --- visualization ---

CLASS_COLORS = {
    'Object':     (0, 255, 0),
    'EmptySlot':  (0, 0, 255),
    'SkuLabel':   (255, 165, 0),
    'Reflection': (255, 165, 0),  # same as SkuLabel visually
    'OfferLabel': (180, 180, 180),
}

ASSOC_COLORS = {
    'Object':    (0, 220, 0),
    'EmptySlot': (50, 50, 255),
}


def draw_associations(
    image: np.ndarray,
    detections: sv.Detections,
    associations: List[dict],
    class_names: Optional[List[str]] = None,
    line_thickness: int = 2,
    box_thickness: int = 2,
    font_scale: float = 0.5,
    dot_radius: int = 5,
) -> np.ndarray:
    if class_names is None:
        class_names = CLASS_NAMES

    vis = image.copy()

    # draw all bounding boxes
    for i, bbox in enumerate(detections.xyxy):
        cls_name = class_names[detections.class_id[i]]
        color = CLASS_COLORS.get(cls_name, (200, 200, 200))
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, box_thickness)

        label = cls_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # draw association lines
    for assoc in associations:
        t_bbox = assoc['target_xyxy']
        l_bbox = assoc['label_xyxy']

        t_cx = int((t_bbox[0] + t_bbox[2]) / 2)
        t_cy = int((t_bbox[1] + t_bbox[3]) / 2)
        l_cx = int((l_bbox[0] + l_bbox[2]) / 2)
        l_cy = int((l_bbox[1] + l_bbox[3]) / 2)

        line_color = ASSOC_COLORS.get(assoc['target_type'], (255, 255, 255))

        cv2.circle(vis, (t_cx, t_cy), dot_radius, line_color, -1)
        cv2.line(vis, (t_cx, t_cy), (l_cx, l_cy), line_color, line_thickness, cv2.LINE_AA)
        cv2.circle(vis, (l_cx, l_cy), dot_radius, line_color, -1)

    return vis