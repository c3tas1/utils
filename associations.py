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


def _horizontal_overlap_ratio(target_bbox, label_bbox):
    """Fraction of the label's width that overlaps horizontally with the target."""
    overlap_left = max(target_bbox[0], label_bbox[0])
    overlap_right = min(target_bbox[2], label_bbox[2])
    overlap = max(0, overlap_right - overlap_left)
    label_width = label_bbox[2] - label_bbox[0]
    if label_width <= 0:
        return 0.0
    return overlap / label_width


def associate_sku_labels(
    detections: sv.Detections,
    img_height: int,
    img_width: int,
    class_names: Optional[List[str]] = None,
    shelf_gap_ratio: float = 0.08,
    alignment_tolerance_ratio: float = 0.03,
    right_side_overlap_threshold: float = 0.80,
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

    targets = []
    sku_labels = []

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

    # cluster labels into shelf rows
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

    def _is_valid_association(item_bbox, lbl_bbox):
        """
        Check association validity:
        1. Label must be below the target (no upward associations)
        2. If label is to the right of target center, require >=80% overlap
        """
        # label center must be below target center
        if cy(lbl_bbox) <= cy(item_bbox):
            return False

        # if label's left edge is to the right of the target's center,
        # only allow if heavy horizontal overlap
        item_center_x = cx(item_bbox)
        if left(lbl_bbox) > item_center_x:
            overlap = _horizontal_overlap_ratio(item_bbox, lbl_bbox)
            if overlap < right_side_overlap_threshold:
                return False

        return True

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
            # row must be below target center
            if row_cy <= cy(item_bbox):
                continue
            dist = row_cy - item_bottom_y
            if dist < -shelf_gap_tol:
                continue
            if abs(dist) < best_dist:
                best_dist = abs(dist)
                best_row = row

        if best_row is None:
            # fallback: closest row that is still below target center
            best_dist = float('inf')
            for row in shelf_rows:
                row_cy = np.mean([cy(b) for _, b in row])
                if row_cy <= cy(item_bbox):
                    continue
                dist = abs(row_cy - item_bottom_y)
                if dist < best_dist:
                    best_dist = dist
                    best_row = row

        if best_row is None:
            continue

        # EmptySlot spanning multiple labels
        if item_cls == 'EmptySlot':
            spanning = []
            for lbl_idx, lbl_bbox in best_row:
                if not _is_valid_association(item_bbox, lbl_bbox):
                    continue
                if (right(lbl_bbox) > item_left_x + align_tol
                        and left(lbl_bbox) < item_right_x - align_tol):
                    spanning.append((lbl_idx, lbl_bbox))

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
            if not _is_valid_association(item_bbox, lbl_bbox):
                continue
            lbl_left = left(lbl_bbox)
            if lbl_left <= item_center_x + align_tol:
                dist = item_center_x - lbl_left
                if dist < best_label_dist:
                    best_label_dist = dist
                    best_label_idx = lbl_idx
                    best_label_bbox = lbl_bbox

        # fallback: closest valid label overall
        if best_label_idx is None:
            best_label_dist = float('inf')
            for lbl_idx, lbl_bbox in best_row:
                if not _is_valid_association(item_bbox, lbl_bbox):
                    continue
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


    def group_associations_by_row(
    associations: List[dict],
    img_height: int,
    shelf_gap_ratio: float = 0.08,
) -> List[List[dict]]:
    """
    Group associations into shelf rows based on their label_xyxy vertical
    position, sorted top-to-bottom by row and left-to-right within each row.

    Args:
        associations: List of association dicts from associate_sku_labels().
        img_height: Original image height.
        shelf_gap_ratio: Vertical tolerance for row clustering.

    Returns:
        List of rows (top to bottom), each row is a list of association
        dicts sorted left to right by label position.
    """
    if not associations:
        return []

    shelf_gap_tol = img_height * shelf_gap_ratio

    def label_cy(assoc):
        bbox = assoc['label_xyxy']
        return (bbox[1] + bbox[3]) / 2.0

    def label_left(assoc):
        return assoc['label_xyxy'][0]

    # sort by label vertical center
    sorted_assocs = sorted(associations, key=label_cy)

    # cluster into rows
    rows = []  # each row: list of assoc dicts
    for assoc in sorted_assocs:
        a_cy = label_cy(assoc)
        placed = False
        for row in rows:
            row_cy = np.mean([label_cy(a) for a in row])
            if abs(a_cy - row_cy) < shelf_gap_tol:
                row.append(assoc)
                placed = True
                break
        if not placed:
            rows.append([assoc])

    # sort rows top to bottom, associations within each row left to right
    rows.sort(key=lambda row: np.mean([label_cy(a) for a in row]))
    for row in rows:
        row.sort(key=label_left)

    return rows