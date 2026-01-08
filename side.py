#!/usr/bin/env python3
"""
Simple pill side detection.
Minimal function to detect if a pill is placed on its side.
"""

import cv2
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict


def load_bboxes(bbox_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Load bboxes from pickle file.
    Returns list of (x, y, w, h) tuples.
    """
    with open(bbox_path, 'rb') as f:
        data = pickle.load(f)
    
    bboxes = []
    for item in data:
        if len(item) >= 4:
            v0, v1, v2, v3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
            
            # Auto-detect format: x1,y1,x2,y2 vs x,y,w,h
            if v2 > v0 and v3 > v1:
                # x1,y1,x2,y2 format
                x, y = int(v0), int(v1)
                w, h = int(v2 - v0), int(v3 - v1)
            else:
                # x,y,w,h format
                x, y, w, h = int(v0), int(v1), int(v2), int(v3)
            
            if w > 0 and h > 0:
                bboxes.append((x, y, w, h))
    
    return bboxes


def segment_pill(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """
    Segment the pill from the background in a cropped bbox region.
    Returns the contour of the pill, or None if failed.
    
    Uses center-anchor: finds the contour that contains the crop center.
    """
    if crop_bgr.shape[0] < 10 or crop_bgr.shape[1] < 10:
        return None
    
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = (w // 2, h // 2)
    
    # Try Otsu thresholding (assumes pill is brighter than tray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up with morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contour containing the center point
    for contour in contours:
        if cv2.pointPolygonTest(contour, center, False) >= 0:
            if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
                return contour
    
    return None


def get_pill_aspect_ratio(contour: np.ndarray) -> float:
    """
    Get the aspect ratio of a pill by fitting an ellipse.
    Returns major_axis / minor_axis (always >= 1.0).
    """
    ellipse = cv2.fitEllipse(contour)
    axes = ellipse[1]  # (width, height) of ellipse
    major = max(axes)
    minor = min(axes)
    
    return major / minor if minor > 0 else 1.0


def is_pill_on_side(crop_bgr: np.ndarray, aspect_threshold: float = 2.5) -> Dict:
    """
    Determine if a pill is placed on its side.
    
    Args:
        crop_bgr: Cropped BGR image containing one pill
        aspect_threshold: Aspect ratio above which pill is considered on-side
    
    Returns:
        Dict with:
            - is_on_side: bool
            - aspect_ratio: float (ellipse major/minor)
            - contour_area: float
            - success: bool (whether segmentation worked)
    """
    result = {
        'is_on_side': False,
        'aspect_ratio': 1.0,
        'contour_area': 0,
        'success': False
    }
    
    # Segment the pill
    contour = segment_pill(crop_bgr)
    
    if contour is None:
        return result
    
    # Get aspect ratio
    aspect_ratio = get_pill_aspect_ratio(contour)
    contour_area = cv2.contourArea(contour)
    
    result['success'] = True
    result['aspect_ratio'] = aspect_ratio
    result['contour_area'] = contour_area
    result['is_on_side'] = aspect_ratio > aspect_threshold
    
    return result


def analyze_image(image_path: str, bbox_path: str, aspect_threshold: float = 2.5) -> List[Dict]:
    """
    Analyze all pills in an image for side placement.
    
    Args:
        image_path: Path to the image
        bbox_path: Path to the bbox pickle file
        aspect_threshold: Aspect ratio threshold for on-side detection
    
    Returns:
        List of results for each pill
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_h, img_w = image.shape[:2]
    
    # Load bboxes
    bboxes = load_bboxes(bbox_path)
    
    results = []
    
    for i, (x, y, w, h) in enumerate(bboxes):
        # Clip to image bounds
        x, y = max(0, x), max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            continue
        
        # Crop
        crop = image[y:y+h, x:x+w]
        
        # Analyze
        pill_result = is_pill_on_side(crop, aspect_threshold)
        pill_result['bbox_id'] = i
        pill_result['bbox'] = (x, y, w, h)
        
        results.append(pill_result)
    
    return results


def visualize_results(image_path: str, bbox_path: str, output_path: str, 
                      aspect_threshold: float = 2.5):
    """
    Visualize side detection results.
    Green = normal, Magenta = on-side, Red = segmentation failed
    """
    image = cv2.imread(image_path)
    bboxes = load_bboxes(bbox_path)
    results = analyze_image(image_path, bbox_path, aspect_threshold)
    
    for r in results:
        x, y, w, h = r['bbox']
        
        if not r['success']:
            color = (0, 0, 255)  # Red - failed
        elif r['is_on_side']:
            color = (255, 0, 255)  # Magenta - on side
        else:
            color = (0, 255, 0)  # Green - normal
        
        thickness = 3 if r['is_on_side'] else 1
        cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        # Show aspect ratio
        label = f"{r['aspect_ratio']:.1f}"
        cv2.putText(image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(output_path, image)
    print(f"Saved: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple pill side detection")
    parser.add_argument('--image', '-i', required=True, help='Image path')
    parser.add_argument('--bboxes', '-b', required=True, help='Bbox pickle file')
    parser.add_argument('--output', '-o', default='./side_detection_result.jpg', help='Output path')
    parser.add_argument('--threshold', '-t', type=float, default=2.5, 
                        help='Aspect ratio threshold (default: 2.5)')
    
    args = parser.parse_args()
    
    # Analyze
    results = analyze_image(args.image, args.bboxes, args.threshold)
    
    # Summary
    total = len(results)
    success = sum(1 for r in results if r['success'])
    on_side = sum(1 for r in results if r['is_on_side'])
    
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Total pills: {total}")
    print(f"Segmented successfully: {success}")
    print(f"Pills on side: {on_side}")
    
    if success > 0:
        aspects = [r['aspect_ratio'] for r in results if r['success']]
        print(f"\nAspect ratios:")
        print(f"  Min: {min(aspects):.2f}")
        print(f"  Max: {max(aspects):.2f}")
        print(f"  Median: {np.median(aspects):.2f}")
        print(f"  Threshold: {args.threshold}")
    
    # Show on-side pills
    if on_side > 0:
        print(f"\nPills on side:")
        for r in results:
            if r['is_on_side']:
                print(f"  #{r['bbox_id']}: aspect={r['aspect_ratio']:.2f}, bbox={r['bbox']}")
    
    # Visualize
    visualize_results(args.image, args.bboxes, args.output, args.threshold)