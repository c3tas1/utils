#!/usr/bin/env python3
"""
Pill Image Quality Assessment v2 - Simplified & Practical
==========================================================

Focuses on detecting:
1. Stacking - pills overlapping each other
2. Pills on side - unusual aspect ratio compared to siblings  
3. Blur - low sharpness (Laplacian variance)
4. Brightness issues - too dark or too bright
5. Motion blur - directional blur from settling pills

Uses simple, tunable thresholds with per-image normalization.
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import cv2
import numpy as np
from tqdm import tqdm

# Force CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''


# =============================================================================
# CONFIGURATION - TUNE THESE THRESHOLDS
# =============================================================================

@dataclass
class QualityThresholds:
    """
    Tunable thresholds for quality detection.
    Adjust these based on your specific dataset.
    """
    # Stacking: IoU threshold (0.0 - 1.0)
    # Higher = more lenient, Lower = more strict
    stacking_iou_threshold: float = 0.15  # 15% overlap = stacking
    
    # Pills on side: aspect ratio threshold
    # Pills with aspect ratio > (median * this multiplier) are flagged
    side_aspect_multiplier: float = 1.8  # 80% more elongated than median
    
    # Pills on side: area threshold  
    # Pills with area < (median * this multiplier) are flagged
    side_area_multiplier: float = 0.5  # Less than 50% of median area
    
    # Blur: Laplacian variance threshold
    # Pills with Laplacian < (median * this multiplier) are flagged
    blur_threshold_multiplier: float = 0.3  # Less than 30% of median sharpness
    
    # Brightness: deviation from median
    # Pills outside this range are flagged
    brightness_low_multiplier: float = 0.6   # Less than 60% of median brightness
    brightness_high_multiplier: float = 1.5  # More than 150% of median brightness
    
    # Image-level blur threshold (absolute Laplacian variance)
    image_blur_threshold: float = 50.0
    
    # Image-level brightness thresholds (0-255)
    image_brightness_low: float = 40.0
    image_brightness_high: float = 220.0
    
    # Overexposure: % of pixels > 250
    overexposure_threshold: float = 0.05  # 5% overexposed pixels
    
    # Underexposure: % of pixels < 10
    underexposure_threshold: float = 0.05  # 5% underexposed pixels


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass 
class PillResult:
    """Result for a single pill."""
    bbox_id: int
    bbox: Tuple[int, int, int, int]
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ImageResult:
    """Result for a single image."""
    image_id: str
    is_bad: bool = False
    image_issues: List[str] = field(default_factory=list)
    pill_issues: List[str] = field(default_factory=list)
    total_pills: int = 0
    bad_pills: int = 0
    pill_results: List[PillResult] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# BBOX LOADING
# =============================================================================

def load_bboxes(bbox_path: str) -> List[Tuple[int, int, int, int]]:
    """Load bboxes from pickle file. Auto-detects x1y1x2y2 vs xywh format."""
    bboxes = []
    
    if not os.path.exists(bbox_path):
        return bboxes
    
    ext = os.path.splitext(bbox_path)[1].lower()
    
    try:
        if ext in ['.pkl', '.pickle']:
            with open(bbox_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, (list, np.ndarray)):
                for item in data:
                    if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 4:
                        v0, v1, v2, v3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                        
                        # Auto-detect: if v2 > v0 and v3 > v1, it's x1,y1,x2,y2
                        if v2 > v0 and v3 > v1:
                            x, y = int(v0), int(v1)
                            w, h = int(v2 - v0), int(v3 - v1)
                        else:
                            x, y, w, h = int(v0), int(v1), int(v2), int(v3)
                        
                        if w > 0 and h > 0:
                            bboxes.append((x, y, w, h))
        
        elif ext == '.json':
            with open(bbox_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (list, tuple)) and len(item) >= 4:
                        bboxes.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
        
        elif ext == '.txt':
            with open(bbox_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        vals = [float(p) for p in parts[-4:]]
                        bboxes.append((int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])))
    
    except Exception as e:
        print(f"Warning: Could not load bboxes from {bbox_path}: {e}")
    
    return bboxes


# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute Laplacian variance (sharpness measure)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Intersection
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_max_overlap(bbox: Tuple[int, int, int, int], 
                        all_bboxes: List[Tuple[int, int, int, int]]) -> float:
    """Compute maximum overlap ratio with any other bbox."""
    x1, y1, w1, h1 = bbox
    area1 = w1 * h1
    if area1 == 0:
        return 0.0
    
    max_overlap = 0.0
    
    for other in all_bboxes:
        if other == bbox:
            continue
        
        x2, y2, w2, h2 = other
        
        # Intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            # Overlap relative to this bbox
            overlap = intersection / area1
            max_overlap = max(max_overlap, overlap)
    
    return max_overlap


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_image(image_path: str, bbox_path: str, 
                  thresholds: QualityThresholds) -> ImageResult:
    """Analyze a single image for quality issues."""
    
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    result = ImageResult(image_id=image_id)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        result.image_issues.append("failed_to_load")
        result.is_bad = True
        return result
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    
    # Load bboxes
    bboxes = load_bboxes(bbox_path)
    result.total_pills = len(bboxes)
    
    # === IMAGE-LEVEL ANALYSIS ===
    
    # Blur
    img_laplacian = compute_laplacian_variance(gray)
    result.metrics['laplacian_variance'] = img_laplacian
    if img_laplacian < thresholds.image_blur_threshold:
        result.image_issues.append(f"image_blur(lap={img_laplacian:.1f})")
    
    # Brightness
    mean_brightness = float(np.mean(gray))
    result.metrics['mean_brightness'] = mean_brightness
    if mean_brightness < thresholds.image_brightness_low:
        result.image_issues.append(f"image_too_dark(brightness={mean_brightness:.1f})")
    elif mean_brightness > thresholds.image_brightness_high:
        result.image_issues.append(f"image_too_bright(brightness={mean_brightness:.1f})")
    
    # Overexposure
    overexposed_ratio = np.sum(gray > 250) / gray.size
    result.metrics['overexposed_ratio'] = overexposed_ratio
    if overexposed_ratio > thresholds.overexposure_threshold:
        result.image_issues.append(f"overexposed({overexposed_ratio*100:.1f}%)")
    
    # Underexposure
    underexposed_ratio = np.sum(gray < 10) / gray.size
    result.metrics['underexposed_ratio'] = underexposed_ratio
    if underexposed_ratio > thresholds.underexposure_threshold:
        result.image_issues.append(f"underexposed({underexposed_ratio*100:.1f}%)")
    
    # === PILL-LEVEL ANALYSIS ===
    
    if not bboxes:
        result.image_issues.append("no_pills_detected")
        result.is_bad = True
        return result
    
    # Compute per-pill metrics first
    pill_metrics = []
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_w - x), min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            continue
        
        crop_gray = gray[y:y+h, x:x+w]
        
        metrics = {
            'bbox_id': i,
            'bbox': (x, y, w, h),
            'area': w * h,
            'aspect_ratio': max(w/h, h/w),  # Always >= 1
            'laplacian': compute_laplacian_variance(crop_gray),
            'brightness': float(np.mean(crop_gray)),
            'overlap': compute_max_overlap(bbox, bboxes),
        }
        pill_metrics.append(metrics)
    
    if not pill_metrics:
        return result
    
    # Compute medians for relative comparisons
    areas = [p['area'] for p in pill_metrics]
    aspects = [p['aspect_ratio'] for p in pill_metrics]
    laplacians = [p['laplacian'] for p in pill_metrics]
    brightnesses = [p['brightness'] for p in pill_metrics]
    
    median_area = np.median(areas)
    median_aspect = np.median(aspects)
    median_laplacian = np.median(laplacians)
    median_brightness = np.median(brightnesses)
    
    result.metrics['median_area'] = median_area
    result.metrics['median_aspect'] = median_aspect
    result.metrics['median_laplacian'] = median_laplacian
    result.metrics['median_brightness'] = median_brightness
    
    # Analyze each pill
    stacking_count = 0
    side_count = 0
    blur_count = 0
    bright_count = 0
    dark_count = 0
    
    for pm in pill_metrics:
        pill_result = PillResult(
            bbox_id=pm['bbox_id'],
            bbox=pm['bbox'],
            metrics=pm
        )
        
        # Check stacking
        if pm['overlap'] > thresholds.stacking_iou_threshold:
            pill_result.issues.append(f"stacking({pm['overlap']*100:.0f}%)")
            stacking_count += 1
        
        # Check on-side (elongated AND small)
        is_elongated = pm['aspect_ratio'] > median_aspect * thresholds.side_aspect_multiplier
        is_small = pm['area'] < median_area * thresholds.side_area_multiplier
        if is_elongated and is_small:
            pill_result.issues.append(f"on_side(aspect={pm['aspect_ratio']:.2f},area_ratio={pm['area']/median_area:.0%})")
            side_count += 1
        
        # Check blur
        if pm['laplacian'] < median_laplacian * thresholds.blur_threshold_multiplier:
            pill_result.issues.append(f"blur(lap={pm['laplacian']:.0f})")
            blur_count += 1
        
        # Check brightness
        if pm['brightness'] < median_brightness * thresholds.brightness_low_multiplier:
            pill_result.issues.append(f"dark(bright={pm['brightness']:.0f})")
            dark_count += 1
        elif pm['brightness'] > median_brightness * thresholds.brightness_high_multiplier:
            pill_result.issues.append(f"bright(bright={pm['brightness']:.0f})")
            bright_count += 1
        
        if pill_result.issues:
            result.bad_pills += 1
        
        result.pill_results.append(pill_result)
    
    # Aggregate pill issues
    if stacking_count > 0:
        result.pill_issues.append(f"stacking:{stacking_count}")
    if side_count > 0:
        result.pill_issues.append(f"on_side:{side_count}")
    if blur_count > 0:
        result.pill_issues.append(f"blur:{blur_count}")
    if dark_count > 0:
        result.pill_issues.append(f"dark:{dark_count}")
    if bright_count > 0:
        result.pill_issues.append(f"bright:{bright_count}")
    
    # Determine if image is bad overall
    result.is_bad = len(result.image_issues) > 0 or result.bad_pills > 0
    
    return result


def analyze_image_wrapper(args):
    """Wrapper for multiprocessing."""
    image_path, bbox_path, thresholds = args
    try:
        return analyze_image(image_path, bbox_path, thresholds)
    except Exception as e:
        result = ImageResult(image_id=os.path.basename(image_path))
        result.image_issues.append(f"error:{str(e)}")
        result.is_bad = True
        return result


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_pairs(images_dir: str, bboxes_dir: str) -> List[Tuple[str, str]]:
    """Match images with bbox files."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    bbox_extensions = {'.pkl', '.pickle', '.json', '.txt'}
    
    # Get all images
    images = {}
    for f in os.listdir(images_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in image_extensions:
            images[name] = os.path.join(images_dir, f)
    
    # Get all bboxes
    bboxes = {}
    for f in os.listdir(bboxes_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in bbox_extensions:
            bboxes[name] = os.path.join(bboxes_dir, f)
    
    # Match
    pairs = []
    for name, img_path in images.items():
        bbox_path = bboxes.get(name, '')
        pairs.append((img_path, bbox_path))
    
    return pairs


# =============================================================================
# DEBUG MODE
# =============================================================================

def debug_image(image_path: str, bbox_path: str, output_dir: str, 
                thresholds: QualityThresholds):
    """Debug a single image with detailed output and visualization."""
    
    print(f"\n{'='*70}")
    print(f"DEBUG: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # Analyze
    result = analyze_image(image_path, bbox_path, thresholds)
    
    # Print metrics
    print(f"\nImage metrics:")
    for k, v in result.metrics.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    
    print(f"\nThresholds being used:")
    print(f"  Stacking: overlap > {thresholds.stacking_iou_threshold*100:.0f}%")
    print(f"  On-side: aspect > median*{thresholds.side_aspect_multiplier} AND area < median*{thresholds.side_area_multiplier}")
    print(f"  Blur: laplacian < median*{thresholds.blur_threshold_multiplier}")
    print(f"  Dark: brightness < median*{thresholds.brightness_low_multiplier}")
    print(f"  Bright: brightness > median*{thresholds.brightness_high_multiplier}")
    
    median_aspect = result.metrics.get('median_aspect', 1.0)
    median_area = result.metrics.get('median_area', 1.0)
    median_lap = result.metrics.get('median_laplacian', 1.0)
    median_bright = result.metrics.get('median_brightness', 1.0)
    
    print(f"\nComputed thresholds for this image:")
    print(f"  On-side aspect threshold: > {median_aspect * thresholds.side_aspect_multiplier:.2f}")
    print(f"  On-side area threshold: < {median_area * thresholds.side_area_multiplier:.0f}")
    print(f"  Blur threshold: < {median_lap * thresholds.blur_threshold_multiplier:.0f}")
    print(f"  Dark threshold: < {median_bright * thresholds.brightness_low_multiplier:.0f}")
    print(f"  Bright threshold: > {median_bright * thresholds.brightness_high_multiplier:.0f}")
    
    print(f"\nImage issues: {result.image_issues}")
    print(f"Pill issues: {result.pill_issues}")
    print(f"Total pills: {result.total_pills}, Bad pills: {result.bad_pills}")
    
    # Show top pills by each metric
    if result.pill_results:
        pills = result.pill_results
        
        print(f"\nTOP 5 by overlap:")
        for p in sorted(pills, key=lambda x: x.metrics.get('overlap', 0), reverse=True)[:5]:
            m = p.metrics
            print(f"  #{m['bbox_id']}: overlap={m['overlap']*100:.1f}%, issues={p.issues}")
        
        print(f"\nTOP 5 by aspect ratio:")
        for p in sorted(pills, key=lambda x: x.metrics.get('aspect_ratio', 0), reverse=True)[:5]:
            m = p.metrics
            print(f"  #{m['bbox_id']}: aspect={m['aspect_ratio']:.2f}, area={m['area']}, issues={p.issues}")
        
        print(f"\nBOTTOM 5 by area:")
        for p in sorted(pills, key=lambda x: x.metrics.get('area', float('inf')))[:5]:
            m = p.metrics
            print(f"  #{m['bbox_id']}: area={m['area']}, aspect={m['aspect_ratio']:.2f}, issues={p.issues}")
        
        print(f"\nBOTTOM 5 by laplacian (blurriest):")
        for p in sorted(pills, key=lambda x: x.metrics.get('laplacian', float('inf')))[:5]:
            m = p.metrics
            print(f"  #{m['bbox_id']}: lap={m['laplacian']:.0f}, issues={p.issues}")
    
    # Create visualization
    image = cv2.imread(image_path)
    if image is not None:
        # Draw legend
        cv2.rectangle(image, (10, 10), (250, 90), (255, 255, 255), -1)
        cv2.putText(image, "Green=OK", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
        cv2.putText(image, "Red=Stacking", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(image, "Magenta=OnSide, Orange=Blur", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 0, 128), 1)
        
        for pr in result.pill_results:
            x, y, w, h = pr.bbox
            
            # Color by issue type
            if any('stacking' in i for i in pr.issues):
                color = (0, 0, 255)  # Red
            elif any('on_side' in i for i in pr.issues):
                color = (255, 0, 255)  # Magenta
            elif any('blur' in i for i in pr.issues):
                color = (0, 165, 255)  # Orange
            elif any('dark' in i or 'bright' in i for i in pr.issues):
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            thickness = 3 if pr.issues else 1
            cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)
        
        out_path = os.path.join(output_dir, f"debug_{os.path.basename(image_path)}")
        cv2.imwrite(out_path, image)
        print(f"\nSaved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pill Quality Assessment v2")
    parser.add_argument('--images', '-i', required=True, help='Images directory')
    parser.add_argument('--bboxes', '-b', required=True, help='Bboxes directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=None, help='Parallel workers')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--debug-images', nargs='*', help='Specific images to debug')
    
    # Threshold overrides
    parser.add_argument('--stacking-threshold', type=float, default=0.15,
                       help='Stacking overlap threshold (default: 0.15)')
    parser.add_argument('--side-aspect', type=float, default=1.8,
                       help='Side detection aspect multiplier (default: 1.8)')
    parser.add_argument('--side-area', type=float, default=0.5,
                       help='Side detection area multiplier (default: 0.5)')
    parser.add_argument('--blur-threshold', type=float, default=0.3,
                       help='Blur threshold multiplier (default: 0.3)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Create thresholds
    thresholds = QualityThresholds(
        stacking_iou_threshold=args.stacking_threshold,
        side_aspect_multiplier=args.side_aspect,
        side_area_multiplier=args.side_area,
        blur_threshold_multiplier=args.blur_threshold,
    )
    
    # Find pairs
    pairs = find_pairs(args.images, args.bboxes)
    print(f"Found {len(pairs)} images")
    
    if args.debug:
        # Debug mode
        if args.debug_images:
            pairs = [(img, bbox) for img, bbox in pairs 
                    if os.path.basename(img) in args.debug_images or
                       os.path.splitext(os.path.basename(img))[0] in args.debug_images]
        else:
            pairs = pairs[:5]
        
        for img, bbox in pairs:
            debug_image(img, bbox, args.output, thresholds)
        return
    
    # Full analysis
    num_workers = args.workers or max(1, mp.cpu_count() - 2)
    
    tasks = [(img, bbox, thresholds) for img, bbox in pairs]
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(analyze_image_wrapper, t): t[0] for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
            results.append(future.result())
    
    # Summary
    bad_count = sum(1 for r in results if r.is_bad)
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total images: {len(results)}")
    print(f"Bad images: {bad_count} ({100*bad_count/len(results):.1f}%)")
    print(f"Good images: {len(results) - bad_count}")
    
    # Count specific issues
    issue_counts = {}
    for r in results:
        for issue in r.image_issues + r.pill_issues:
            key = issue.split(':')[0].split('(')[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
    
    print(f"\nIssue breakdown:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count}")
    
    # Save results
    output_data = []
    for r in results:
        output_data.append({
            'image_id': r.image_id,
            'is_bad': r.is_bad,
            'image_issues': r.image_issues,
            'pill_issues': r.pill_issues,
            'total_pills': r.total_pills,
            'bad_pills': r.bad_pills,
        })
    
    output_path = os.path.join(args.output, 'quality_results.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()