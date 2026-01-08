#!/usr/bin/env python3
"""
Pill stacking detection v3.

Key insight: When YOLO detects both stacked pills:
- Top pill bbox = contains normal single pill
- Bottom pill bbox = contains PART of top pill + visible part of bottom pill

So if we segment a bbox and find it contains portions of 2 pills 
(2 separate bright regions), that indicates stacking.

Also: Top pill may cast a SHADOW on the bottom pill, creating a darker
region/edge that's unusual.
"""

import cv2
import numpy as np
import pickle
from typing import List, Tuple, Dict


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


def count_pill_regions(crop_bgr: np.ndarray, min_region_ratio: float = 0.15) -> Dict:
    """
    Count how many separate pill-like regions are in the crop.
    
    If there are 2+ regions, it suggests the bbox contains parts of 
    multiple pills (stacking scenario).
    
    Args:
        crop_bgr: Cropped image
        min_region_ratio: Minimum size of region relative to largest (0-1)
    
    Returns:
        Dict with num_regions, region_areas, etc.
    """
    if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return {'num_regions': 0, 'regions': []}
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to separate touching regions slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    
    # Filter regions (skip background label 0)
    regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 50:  # Minimum pixel area
            regions.append({
                'label': i,
                'area': area,
                'centroid': centroids[i],
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            })
    
    if not regions:
        return {'num_regions': 0, 'regions': []}
    
    # Sort by area
    regions.sort(key=lambda r: r['area'], reverse=True)
    
    # Count significant regions (at least min_region_ratio of the largest)
    largest_area = regions[0]['area']
    significant_regions = [r for r in regions if r['area'] >= largest_area * min_region_ratio]
    
    return {
        'num_regions': len(significant_regions),
        'regions': significant_regions,
        'all_regions': regions
    }


def detect_shadow_edge(crop_bgr: np.ndarray) -> Dict:
    """
    Detect if there's a shadow edge inside the pill region.
    
    When pill A is on top of pill B, pill A casts a shadow creating
    a dark edge/line across pill B's surface.
    
    Returns:
        Dict with has_shadow_edge, shadow_strength
    """
    if crop_bgr.shape[0] < 15 or crop_bgr.shape[1] < 15:
        return {'has_shadow_edge': False, 'shadow_strength': 0}
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Get pill mask
    _, pill_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Look for dark lines/edges within the pill region
    # Use Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Only consider edges inside the pill
    edges_in_pill = cv2.bitwise_and(edges, pill_mask)
    
    # For shadow detection: look for horizontal/vertical dark gradients
    # Shadow creates a distinct brightness change
    
    # Compute gradient
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Strong gradient inside pill = potential shadow edge
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_in_pill = grad_mag * (pill_mask > 0)
    
    # Compute ratio of high-gradient pixels inside pill
    pill_pixels = np.sum(pill_mask > 0)
    if pill_pixels == 0:
        return {'has_shadow_edge': False, 'shadow_strength': 0}
    
    high_grad_pixels = np.sum(grad_in_pill > np.percentile(grad_in_pill[pill_mask > 0], 90))
    grad_ratio = high_grad_pixels / pill_pixels
    
    # Also check for internal edges
    internal_edge_ratio = np.sum(edges_in_pill > 0) / pill_pixels
    
    # Shadow indicator: significant internal gradient/edges
    shadow_strength = (grad_ratio + internal_edge_ratio) / 2
    has_shadow = shadow_strength > 0.05  # Threshold
    
    return {
        'has_shadow_edge': has_shadow,
        'shadow_strength': shadow_strength,
        'internal_edge_ratio': internal_edge_ratio,
        'grad_ratio': grad_ratio
    }


def detect_brightness_discontinuity(crop_bgr: np.ndarray) -> Dict:
    """
    Detect if the pill has a brightness discontinuity.
    
    When stacked, the visible portion of bottom pill may have different
    lighting than the top pill portion that intrudes into the bbox.
    
    Returns:
        Dict with has_discontinuity, brightness_variance
    """
    if crop_bgr.shape[0] < 15 or crop_bgr.shape[1] < 15:
        return {'has_discontinuity': False, 'brightness_std': 0}
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    
    # Get pill mask
    _, pill_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Get brightness values inside pill
    pill_brightness = gray[pill_mask > 0]
    
    if len(pill_brightness) < 10:
        return {'has_discontinuity': False, 'brightness_std': 0}
    
    # Compute statistics
    mean_b = np.mean(pill_brightness)
    std_b = np.std(pill_brightness)
    
    # High std relative to mean = discontinuity
    cv = std_b / mean_b if mean_b > 0 else 0  # Coefficient of variation
    
    # Also check for bimodal distribution (two brightness peaks)
    hist, _ = np.histogram(pill_brightness, bins=20)
    
    # Find peaks in histogram
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(pill_brightness) * 0.05:
            peaks.append(i)
    
    has_discontinuity = cv > 0.15 or len(peaks) >= 2
    
    return {
        'has_discontinuity': has_discontinuity,
        'brightness_std': std_b,
        'brightness_cv': cv,
        'num_brightness_peaks': len(peaks)
    }


def is_stacking(crop_bgr: np.ndarray) -> Dict:
    """
    Determine if a pill bbox shows signs of stacking.
    
    Returns:
        Dict with is_stacking, reason, and detailed metrics
    """
    result = {
        'is_stacking': False,
        'reason': None,
        'confidence': 0.0
    }
    
    # Method 1: Multiple regions
    regions = count_pill_regions(crop_bgr)
    result['num_regions'] = regions['num_regions']
    
    if regions['num_regions'] >= 2:
        result['is_stacking'] = True
        result['reason'] = f"multiple_regions({regions['num_regions']})"
        result['confidence'] = 0.9
        return result
    
    # Method 2: Shadow edge detection
    shadow = detect_shadow_edge(crop_bgr)
    result['shadow_strength'] = shadow['shadow_strength']
    
    if shadow['has_shadow_edge'] and shadow['shadow_strength'] > 0.08:
        result['is_stacking'] = True
        result['reason'] = f"shadow_edge({shadow['shadow_strength']:.2f})"
        result['confidence'] = 0.7
        return result
    
    # Method 3: Brightness discontinuity
    brightness = detect_brightness_discontinuity(crop_bgr)
    result['brightness_cv'] = brightness['brightness_cv']
    result['num_brightness_peaks'] = brightness['num_brightness_peaks']
    
    if brightness['has_discontinuity'] and brightness['num_brightness_peaks'] >= 2:
        result['is_stacking'] = True
        result['reason'] = f"brightness_discontinuity(peaks={brightness['num_brightness_peaks']})"
        result['confidence'] = 0.6
        return result
    
    return result


def analyze_image(image_path: str, bbox_path: str) -> Dict:
    """Analyze all pills for stacking."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load: {image_path}")
    
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
        analysis = is_stacking(crop)
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


def visualize(image_path: str, bbox_path: str, output_path: str):
    """Visualize stacking detection."""
    image = cv2.imread(image_path)
    result = analyze_image(image_path, bbox_path)
    
    for pill in result['pills']:
        x, y, w, h = pill['bbox']
        
        if pill['is_stacking']:
            color = (0, 0, 255)  # Red
            thickness = 3
        else:
            color = (0, 255, 0)  # Green
            thickness = 1
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
    
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', required=True)
    parser.add_argument('--bboxes', '-b', required=True)
    parser.add_argument('--output', '-o', default='./stacking_v3_result.jpg')
    
    args = parser.parse_args()
    
    result = analyze_image(args.image, args.bboxes)
    
    print(f"\nTotal pills: {result['total_pills']}")
    print(f"Stacking detected: {result['stacking_count']}")
    
    stacked = [p for p in result['pills'] if p['is_stacking']]
    if stacked:
        print(f"\nStacked pills:")
        for p in stacked:
            print(f"  #{p['bbox_id']}: {p['reason']}")
    
    visualize(args.image, args.bboxes, args.output)