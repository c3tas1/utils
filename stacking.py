#!/usr/bin/env python3
"""
Pill stacking detection v2.
Detects when a pill is stacked on another by analyzing the segmented shape.

Key insight: When stacked, the bottom pill's contour is:
- Irregular (not a clean ellipse)
- Partially occluded (missing part of its shape)
- Has poor ellipse fit quality

Detection methods:
1. Ellipse fit quality - how well does contour match fitted ellipse
2. Contour solidity - ratio of contour area to convex hull area
3. Multiple contours - if bbox contains multiple separate pill regions
"""

import cv2
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional


def load_bboxes(bbox_path: str) -> List[Tuple[int, int, int, int]]:
    """Load bboxes from pickle file."""
    with open(bbox_path, 'rb') as f:
        data = pickle.load(f)
    
    bboxes = []
    for item in data:
        if len(item) >= 4:
            v0, v1, v2, v3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            if v2 > v0 and v3 > v1:
                x, y = int(v0), int(v1)
                w, h = int(v2 - v0), int(v3 - v1)
            else:
                x, y, w, h = int(v0), int(v1), int(v2), int(v3)
            if w > 0 and h > 0:
                bboxes.append((x, y, w, h))
    return bboxes


def segment_all_contours(crop_bgr: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
    """
    Segment ALL contours in a crop (not just the center one).
    Returns list of contours.
    """
    if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return []
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    return valid_contours


def compute_ellipse_fit_quality(contour: np.ndarray) -> float:
    """
    Compute how well a contour fits an ellipse.
    
    Returns value 0-1:
    - 1.0 = perfect ellipse
    - <0.8 = irregular shape (possible occlusion)
    """
    if len(contour) < 5:
        return 0.0
    
    # Fit ellipse
    ellipse = cv2.fitEllipse(contour)
    
    # Create ellipse mask
    h = int(ellipse[0][1] + ellipse[1][1]) + 10
    w = int(ellipse[0][0] + ellipse[1][0]) + 10
    h = max(h, contour[:, :, 1].max() + 10)
    w = max(w, contour[:, :, 0].max() + 10)
    
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)
    
    # Create contour mask
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
    
    # Compute overlap
    intersection = np.sum((ellipse_mask > 0) & (contour_mask > 0))
    union = np.sum((ellipse_mask > 0) | (contour_mask > 0))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_solidity(contour: np.ndarray) -> float:
    """
    Compute solidity = contour_area / convex_hull_area.
    
    Low solidity = irregular/concave shape (possible occlusion)
    Normal pill should have high solidity (~0.95+)
    """
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0.0
    
    return area / hull_area


def compute_circularity(contour: np.ndarray) -> float:
    """
    Compute circularity = 4π * area / perimeter²
    
    1.0 = perfect circle
    Lower values = irregular shape
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0.0
    
    return 4 * np.pi * area / (perimeter ** 2)


def analyze_pill_for_stacking(crop_bgr: np.ndarray) -> Dict:
    """
    Analyze a single pill crop for signs of stacking.
    
    Returns dict with:
    - is_stacking: bool - whether stacking is detected
    - num_contours: int - number of pill-like regions found
    - ellipse_fit: float - how well main contour fits ellipse (0-1)
    - solidity: float - contour area / convex hull area (0-1)
    - circularity: float - shape circularity (0-1)
    """
    result = {
        'is_stacking': False,
        'num_contours': 0,
        'ellipse_fit': 1.0,
        'solidity': 1.0,
        'circularity': 1.0,
        'reason': None
    }
    
    # Get all contours in the crop
    contours = segment_all_contours(crop_bgr)
    result['num_contours'] = len(contours)
    
    if len(contours) == 0:
        return result
    
    # Method 1: Multiple large contours = likely stacking
    # If we see 2+ distinct pill-shaped regions, pills are stacked
    crop_area = crop_bgr.shape[0] * crop_bgr.shape[1]
    large_contours = [c for c in contours if cv2.contourArea(c) > crop_area * 0.1]
    
    if len(large_contours) >= 2:
        result['is_stacking'] = True
        result['reason'] = f'multiple_contours({len(large_contours)})'
        return result
    
    # Method 2: Analyze the main (largest) contour
    main_contour = max(contours, key=cv2.contourArea)
    
    if len(main_contour) >= 5:
        # Ellipse fit quality
        ellipse_fit = compute_ellipse_fit_quality(main_contour)
        result['ellipse_fit'] = ellipse_fit
        
        # Solidity
        solidity = compute_solidity(main_contour)
        result['solidity'] = solidity
        
        # Circularity
        circularity = compute_circularity(main_contour)
        result['circularity'] = circularity
        
        # Stacking indicators:
        # - Poor ellipse fit (contour doesn't match ellipse shape)
        # - Low solidity (concave/irregular shape from occlusion)
        
        if ellipse_fit < 0.75:
            result['is_stacking'] = True
            result['reason'] = f'poor_ellipse_fit({ellipse_fit:.2f})'
        elif solidity < 0.85:
            result['is_stacking'] = True
            result['reason'] = f'low_solidity({solidity:.2f})'
    
    return result


def analyze_image(image_path: str, bbox_path: str) -> Dict:
    """Analyze all pills in an image for stacking."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_h, img_w = image.shape[:2]
    bboxes = load_bboxes(bbox_path)
    
    results = []
    stacking_count = 0
    
    for i, (x, y, w, h) in enumerate(bboxes):
        x, y = max(0, x), max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            continue
        
        crop = image[y:y+h, x:x+w]
        
        analysis = analyze_pill_for_stacking(crop)
        analysis['bbox_id'] = i
        analysis['bbox'] = (x, y, w, h)
        
        if analysis['is_stacking']:
            stacking_count += 1
        
        results.append(analysis)
    
    return {
        'total_pills': len(bboxes),
        'stacking_count': stacking_count,
        'pills': results
    }


def visualize_stacking(image_path: str, bbox_path: str, output_path: str):
    """
    Visualize stacking detection.
    Green = normal, Red = stacking detected
    """
    image = cv2.imread(image_path)
    result = analyze_image(image_path, bbox_path)
    
    for pill in result['pills']:
        x, y, w, h = pill['bbox']
        
        if pill['is_stacking']:
            color = (0, 0, 255)  # Red
            thickness = 3
            # Show reason
            label = pill['reason'][:15] if pill['reason'] else 'stack'
            cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        else:
            color = (0, 255, 0)  # Green
            thickness = 1
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    
    # Legend
    cv2.rectangle(image, (10, 10), (200, 50), (255, 255, 255), -1)
    cv2.putText(image, "Green=OK, Red=Stacking", (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Pill stacking detection v2")
    parser.add_argument('--image', '-i', required=True, help='Image path')
    parser.add_argument('--bboxes', '-b', required=True, help='Bbox pickle file')
    parser.add_argument('--output', '-o', default='./stacking_v2_result.jpg')
    
    args = parser.parse_args()
    
    # Analyze
    result = analyze_image(args.image, args.bboxes)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"STACKING DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Total pills: {result['total_pills']}")
    print(f"Stacking detected: {result['stacking_count']}")
    
    # Show stacking details
    stacked = [p for p in result['pills'] if p['is_stacking']]
    if stacked:
        print(f"\nStacked pills:")
        for p in stacked:
            print(f"  #{p['bbox_id']}: {p['reason']}")
            print(f"    ellipse_fit={p['ellipse_fit']:.2f}, "
                  f"solidity={p['solidity']:.2f}, "
                  f"contours={p['num_contours']}")
    
    # Show metrics distribution
    pills = result['pills']
    if pills:
        fits = [p['ellipse_fit'] for p in pills]
        sols = [p['solidity'] for p in pills]
        
        print(f"\nMetrics distribution:")
        print(f"  Ellipse fit: min={min(fits):.2f}, max={max(fits):.2f}, median={np.median(fits):.2f}")
        print(f"  Solidity: min={min(sols):.2f}, max={max(sols):.2f}, median={np.median(sols):.2f}")
    
    # Visualize
    visualize_stacking(args.image, args.bboxes, args.output)