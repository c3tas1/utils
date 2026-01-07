#!/usr/bin/env python3
"""
Pill Image Quality Assessment System
=====================================
Traditional CV-based quality classifier for pharmaceutical pill verification.

Two-phase operation:
1. Calibration: Analyze dataset to establish thresholds (worst 15%)
2. Inference: Classify images and pills based on calibrated thresholds

Author: Claude
Date: 2025
"""

import os
import json
import pickle
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import cv2
import numpy as np
from scipy import ndimage, fftpack
from scipy.stats import entropy
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class QualityMetrics:
    """Container for all computed quality metrics."""
    # Blur metrics
    laplacian_variance: float = 0.0
    tenengrad: float = 0.0
    fft_high_freq_ratio: float = 0.0
    brenner_gradient: float = 0.0
    
    # Motion blur specific
    motion_blur_score: float = 0.0
    motion_blur_angle: float = 0.0
    
    # Brightness/exposure metrics
    mean_brightness: float = 0.0
    brightness_std: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    dynamic_range: float = 0.0
    
    # Contrast metrics
    michelson_contrast: float = 0.0
    rms_contrast: float = 0.0
    
    # Local variation metrics (for detecting partial issues)
    brightness_uniformity: float = 0.0
    blur_uniformity: float = 0.0
    
    # Light intrusion detection
    saturation_peaks: float = 0.0
    high_saturation_ratio: float = 0.0
    light_blob_score: float = 0.0
    
    # Histogram-based metrics
    histogram_entropy: float = 0.0
    histogram_skewness: float = 0.0


@dataclass
class PillMetrics:
    """Metrics for individual pill crops."""
    bbox_id: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    
    # Basic metrics
    laplacian_variance: float = 0.0
    tenengrad: float = 0.0
    mean_brightness: float = 0.0
    overexposed_ratio: float = 0.0
    underexposed_ratio: float = 0.0
    
    # Stacking detection
    overlap_ratio: float = 0.0
    aspect_ratio: float = 0.0
    area_deviation: float = 0.0  # deviation from median pill area
    
    # Edge quality
    edge_density: float = 0.0
    edge_continuity: float = 0.0
    
    # Motion blur
    motion_blur_score: float = 0.0
    
    # Issues identified
    issues: List[str] = field(default_factory=list)


@dataclass
class ImageAnalysisResult:
    """Complete analysis result for an image."""
    image_id: str = ""
    image_path: str = ""
    is_bad: bool = False
    quality_score: float = 1.0
    image_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    image_level_issues: List[str] = field(default_factory=list)
    pill_level_issues: List[str] = field(default_factory=list)
    pill_metrics: List[PillMetrics] = field(default_factory=list)
    bad_pill_count: int = 0
    total_pill_count: int = 0


@dataclass
class CalibrationThresholds:
    """Thresholds determined during calibration phase."""
    # Blur thresholds (lower = worse)
    laplacian_variance_threshold: float = 0.0
    tenengrad_threshold: float = 0.0
    fft_high_freq_threshold: float = 0.0
    
    # Brightness thresholds
    brightness_low_threshold: float = 0.0
    brightness_high_threshold: float = 0.0
    overexposed_threshold: float = 0.0
    underexposed_threshold: float = 0.0
    
    # Uniformity thresholds
    brightness_uniformity_threshold: float = 0.0
    blur_uniformity_threshold: float = 0.0
    
    # Light intrusion
    saturation_peaks_threshold: float = 0.0
    light_blob_threshold: float = 0.0
    
    # Motion blur
    motion_blur_threshold: float = 0.0
    
    # Pill-level thresholds
    pill_blur_threshold: float = 0.0
    pill_brightness_low: float = 0.0
    pill_brightness_high: float = 0.0
    pill_overlap_threshold: float = 0.0
    pill_aspect_ratio_threshold: float = 0.0
    pill_area_deviation_threshold: float = 0.0
    
    # Contrast
    contrast_threshold: float = 0.0
    
    # Statistics from calibration
    calibration_stats: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# IMAGE QUALITY METRICS COMPUTATION
# =============================================================================

class QualityMetricsComputer:
    """Compute various image quality metrics using traditional CV algorithms."""
    
    def __init__(self):
        pass
    
    @staticmethod
    def compute_laplacian_variance(gray: np.ndarray) -> float:
        """
        Laplacian variance - classic blur detection.
        Lower values indicate more blur.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())
    
    @staticmethod
    def compute_tenengrad(gray: np.ndarray) -> float:
        """
        Tenengrad focus measure using Sobel gradients.
        More robust to noise than Laplacian.
        """
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        return float(np.mean(gradient_magnitude**2))
    
    @staticmethod
    def compute_brenner_gradient(gray: np.ndarray) -> float:
        """
        Brenner's focus measure - difference between pixels 2 apart.
        """
        diff = gray[:, 2:].astype(np.float64) - gray[:, :-2].astype(np.float64)
        return float(np.mean(diff**2))
    
    @staticmethod
    def compute_fft_blur_metric(gray: np.ndarray) -> float:
        """
        FFT-based blur detection.
        Analyzes high-frequency content ratio.
        """
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Compute FFT
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Create masks for low and high frequencies
        # Low frequency: center 10% of the spectrum
        low_freq_radius = min(rows, cols) // 20
        y, x = np.ogrid[:rows, :cols]
        low_mask = ((x - ccol)**2 + (y - crow)**2) <= low_freq_radius**2
        
        # High frequency: outer 50% of the spectrum
        high_freq_radius = min(rows, cols) // 4
        high_mask = ((x - ccol)**2 + (y - crow)**2) >= high_freq_radius**2
        
        low_energy = np.sum(magnitude[low_mask])
        high_energy = np.sum(magnitude[high_mask])
        
        if low_energy > 0:
            return float(high_energy / (low_energy + 1e-10))
        return 0.0
    
    @staticmethod
    def compute_motion_blur_metrics(gray: np.ndarray) -> Tuple[float, float]:
        """
        Detect motion blur using directional FFT analysis.
        Returns (motion_blur_score, dominant_angle).
        """
        rows, cols = gray.shape
        
        # Compute FFT magnitude spectrum
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        
        # Analyze directional energy
        center = (rows // 2, cols // 2)
        angles = np.linspace(0, 180, 36)  # 5-degree increments
        directional_energy = []
        
        for angle in angles:
            # Create line mask at this angle
            mask = np.zeros((rows, cols), dtype=np.float64)
            length = min(rows, cols) // 2
            
            rad = np.radians(angle)
            for r in range(-length, length):
                x = int(center[1] + r * np.cos(rad))
                y = int(center[0] + r * np.sin(rad))
                if 0 <= x < cols and 0 <= y < rows:
                    # Use a thick line for robustness
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < cols and 0 <= ny < rows:
                                mask[ny, nx] = 1.0
            
            energy = np.sum(magnitude * mask) / (np.sum(mask) + 1e-10)
            directional_energy.append(energy)
        
        directional_energy = np.array(directional_energy)
        
        # Motion blur creates a strong line in FFT perpendicular to motion
        max_energy = np.max(directional_energy)
        min_energy = np.min(directional_energy)
        mean_energy = np.mean(directional_energy)
        
        # Anisotropy ratio indicates motion blur
        if mean_energy > 0:
            anisotropy = (max_energy - min_energy) / (mean_energy + 1e-10)
        else:
            anisotropy = 0.0
        
        dominant_angle = angles[np.argmax(directional_energy)]
        
        return float(anisotropy), float(dominant_angle)
    
    @staticmethod
    def compute_brightness_metrics(gray: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive brightness and exposure analysis.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_normalized = hist / hist.sum()
        
        mean_brightness = float(np.mean(gray))
        brightness_std = float(np.std(gray))
        
        # Overexposed: pixels > 250
        overexposed_ratio = float(np.sum(gray > 250) / gray.size)
        
        # Underexposed: pixels < 5
        underexposed_ratio = float(np.sum(gray < 5) / gray.size)
        
        # Dynamic range
        p1, p99 = np.percentile(gray, [1, 99])
        dynamic_range = float(p99 - p1)
        
        # Histogram entropy
        hist_entropy = float(entropy(hist_normalized + 1e-10))
        
        # Histogram skewness
        pixel_values = np.arange(256)
        mean_val = np.sum(pixel_values * hist_normalized)
        variance = np.sum(((pixel_values - mean_val) ** 2) * hist_normalized)
        std_val = np.sqrt(variance) + 1e-10
        skewness = np.sum(((pixel_values - mean_val) ** 3) * hist_normalized) / (std_val ** 3)
        
        return {
            'mean_brightness': mean_brightness,
            'brightness_std': brightness_std,
            'overexposed_ratio': overexposed_ratio,
            'underexposed_ratio': underexposed_ratio,
            'dynamic_range': dynamic_range,
            'histogram_entropy': hist_entropy,
            'histogram_skewness': float(skewness)
        }
    
    @staticmethod
    def compute_contrast_metrics(gray: np.ndarray) -> Dict[str, float]:
        """
        Compute contrast metrics.
        """
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        
        # Michelson contrast
        if max_val + min_val > 0:
            michelson = (max_val - min_val) / (max_val + min_val)
        else:
            michelson = 0.0
        
        # RMS contrast
        mean_val = np.mean(gray)
        rms_contrast = float(np.sqrt(np.mean((gray.astype(np.float64) - mean_val) ** 2)) / 255.0)
        
        return {
            'michelson_contrast': float(michelson),
            'rms_contrast': rms_contrast
        }
    
    @staticmethod
    def compute_local_uniformity(gray: np.ndarray, grid_size: int = 5) -> Dict[str, float]:
        """
        Analyze local uniformity by dividing image into grid.
        Helps detect partial blur or brightness issues.
        """
        h, w = gray.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        brightness_values = []
        blur_values = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell = gray[y1:y2, x1:x2]
                
                if cell.size > 0:
                    brightness_values.append(np.mean(cell))
                    blur_values.append(cv2.Laplacian(cell, cv2.CV_64F).var())
        
        brightness_values = np.array(brightness_values)
        blur_values = np.array(blur_values)
        
        # Uniformity = 1 - coefficient of variation
        # Lower uniformity means more variation (partial issues)
        brightness_cv = np.std(brightness_values) / (np.mean(brightness_values) + 1e-10)
        blur_cv = np.std(blur_values) / (np.mean(blur_values) + 1e-10)
        
        return {
            'brightness_uniformity': float(1.0 - min(brightness_cv, 1.0)),
            'blur_uniformity': float(1.0 - min(blur_cv, 1.0))
        }
    
    @staticmethod
    def compute_light_intrusion_metrics(image: np.ndarray) -> Dict[str, float]:
        """
        Detect external light intrusion using saturation and blob analysis.
        """
        # Convert to HSV
        if len(image.shape) == 2:
            # Grayscale - limited analysis
            gray = image
            hsv = None
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        results = {
            'saturation_peaks': 0.0,
            'high_saturation_ratio': 0.0,
            'light_blob_score': 0.0
        }
        
        if hsv is not None:
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # High saturation with high value often indicates external light
            high_sat_mask = (saturation > 150) & (value > 200)
            results['high_saturation_ratio'] = float(np.sum(high_sat_mask) / saturation.size)
            
            # Peak saturation regions
            results['saturation_peaks'] = float(np.percentile(saturation, 99) / 255.0)
        
        # Detect bright blobs that might be light intrusion
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Calculate total area of bright blobs
            total_blob_area = sum(cv2.contourArea(c) for c in contours)
            results['light_blob_score'] = float(total_blob_area / gray.size)
        
        return results
    
    def compute_all_metrics(self, image: np.ndarray) -> QualityMetrics:
        """
        Compute all quality metrics for an image.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        metrics = QualityMetrics()
        
        # Blur metrics
        metrics.laplacian_variance = self.compute_laplacian_variance(gray)
        metrics.tenengrad = self.compute_tenengrad(gray)
        metrics.fft_high_freq_ratio = self.compute_fft_blur_metric(gray)
        metrics.brenner_gradient = self.compute_brenner_gradient(gray)
        
        # Motion blur
        motion_score, motion_angle = self.compute_motion_blur_metrics(gray)
        metrics.motion_blur_score = motion_score
        metrics.motion_blur_angle = motion_angle
        
        # Brightness
        brightness = self.compute_brightness_metrics(gray)
        metrics.mean_brightness = brightness['mean_brightness']
        metrics.brightness_std = brightness['brightness_std']
        metrics.overexposed_ratio = brightness['overexposed_ratio']
        metrics.underexposed_ratio = brightness['underexposed_ratio']
        metrics.dynamic_range = brightness['dynamic_range']
        metrics.histogram_entropy = brightness['histogram_entropy']
        metrics.histogram_skewness = brightness['histogram_skewness']
        
        # Contrast
        contrast = self.compute_contrast_metrics(gray)
        metrics.michelson_contrast = contrast['michelson_contrast']
        metrics.rms_contrast = contrast['rms_contrast']
        
        # Local uniformity
        uniformity = self.compute_local_uniformity(gray)
        metrics.brightness_uniformity = uniformity['brightness_uniformity']
        metrics.blur_uniformity = uniformity['blur_uniformity']
        
        # Light intrusion
        light = self.compute_light_intrusion_metrics(image)
        metrics.saturation_peaks = light['saturation_peaks']
        metrics.high_saturation_ratio = light['high_saturation_ratio']
        metrics.light_blob_score = light['light_blob_score']
        
        return metrics


# =============================================================================
# PILL-LEVEL ANALYSIS
# =============================================================================

class PillAnalyzer:
    """Analyze individual pills from bounding boxes."""
    
    def __init__(self):
        self.metrics_computer = QualityMetricsComputer()
    
    def compute_overlap_ratio(self, bbox: Tuple[int, int, int, int], 
                             all_bboxes: List[Tuple[int, int, int, int]]) -> float:
        """
        Compute IoU-based overlap ratio with other bboxes.
        High overlap suggests stacking.
        """
        x1, y1, w1, h1 = bbox
        box1_area = w1 * h1
        
        max_overlap = 0.0
        
        for other in all_bboxes:
            if other == bbox:
                continue
            
            x2, y2, w2, h2 = other
            
            # Compute intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                # Use overlap ratio relative to current box
                overlap = intersection / box1_area
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def analyze_pill(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                    bbox_id: int, all_bboxes: List[Tuple[int, int, int, int]],
                    median_area: float) -> PillMetrics:
        """
        Analyze a single pill crop.
        """
        x, y, w, h = bbox
        
        # Ensure valid crop region
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            return PillMetrics(bbox_id=bbox_id, bbox=bbox)
        
        crop = image[y:y+h, x:x+w]
        
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        
        metrics = PillMetrics(bbox_id=bbox_id, bbox=bbox)
        
        # Basic blur metrics
        metrics.laplacian_variance = self.metrics_computer.compute_laplacian_variance(gray)
        metrics.tenengrad = self.metrics_computer.compute_tenengrad(gray)
        
        # Brightness
        metrics.mean_brightness = float(np.mean(gray))
        metrics.overexposed_ratio = float(np.sum(gray > 250) / gray.size)
        metrics.underexposed_ratio = float(np.sum(gray < 5) / gray.size)
        
        # Stacking detection
        metrics.overlap_ratio = self.compute_overlap_ratio(bbox, all_bboxes)
        metrics.aspect_ratio = w / h if h > 0 else 0
        
        area = w * h
        if median_area > 0:
            metrics.area_deviation = abs(area - median_area) / median_area
        
        # Edge analysis for partial occlusion detection
        edges = cv2.Canny(gray, 50, 150)
        metrics.edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Edge continuity - check if edges form complete contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            metrics.edge_continuity = contour_area / (w * h) if w * h > 0 else 0
        
        # Motion blur for this pill
        if gray.shape[0] > 10 and gray.shape[1] > 10:
            motion_score, _ = self.metrics_computer.compute_motion_blur_metrics(gray)
            metrics.motion_blur_score = motion_score
        
        return metrics


# =============================================================================
# BBOX LOADING
# =============================================================================

def load_bboxes(bbox_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Load bounding boxes from various formats.
    Supports: JSON, YOLO format txt, CSV
    Returns list of (x, y, w, h) tuples in pixel coordinates.
    """
    bboxes = []
    
    if not os.path.exists(bbox_path):
        return bboxes
    
    ext = os.path.splitext(bbox_path)[1].lower()
    
    try:
        if ext == '.json':
            with open(bbox_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Format: {"x": x, "y": y, "w": w, "h": h} or {"bbox": [x,y,w,h]}
                        if 'bbox' in item:
                            bbox = item['bbox']
                            bboxes.append((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
                        elif 'x' in item:
                            bboxes.append((int(item['x']), int(item['y']), 
                                          int(item['w']), int(item['h'])))
                        elif 'x1' in item:
                            # x1, y1, x2, y2 format
                            x1, y1 = int(item['x1']), int(item['y1'])
                            x2, y2 = int(item['x2']), int(item['y2'])
                            bboxes.append((x1, y1, x2-x1, y2-y1))
                    elif isinstance(item, (list, tuple)) and len(item) >= 4:
                        bboxes.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
            elif isinstance(data, dict):
                # Handle nested formats
                if 'boxes' in data:
                    for box in data['boxes']:
                        if len(box) >= 4:
                            bboxes.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
                elif 'detections' in data:
                    for det in data['detections']:
                        if 'bbox' in det:
                            b = det['bbox']
                            bboxes.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        
        elif ext == '.txt':
            # Try to detect format: YOLO or plain
            with open(bbox_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    # Check if YOLO format (normalized) or absolute
                    values = [float(p) for p in parts[-4:]]  # Last 4 values
                    
                    # YOLO format: class x_center y_center width height (normalized)
                    if all(0 <= v <= 1 for v in values):
                        # This is likely YOLO format - we need image dimensions
                        # For now, store as normalized and convert later
                        # Actually, we can't convert without image dims, so assume absolute
                        pass
                    
                    # Assume absolute pixel coordinates: x y w h or x1 y1 x2 y2
                    if len(parts) == 5:
                        # YOLO with class: class x_center y_center w h
                        # Assume these need conversion if normalized
                        values = [float(p) for p in parts[1:5]]
                        if all(0 <= v <= 1 for v in values):
                            # Normalized - mark for later conversion
                            bboxes.append(('yolo', values[0], values[1], values[2], values[3]))
                        else:
                            # Absolute center format
                            cx, cy, w, h = values
                            bboxes.append((int(cx - w/2), int(cy - h/2), int(w), int(h)))
                    elif len(parts) == 4:
                        values = [float(p) for p in parts]
                        if all(0 <= v <= 1 for v in values):
                            bboxes.append(('yolo', values[0], values[1], values[2], values[3]))
                        else:
                            # x, y, w, h absolute
                            bboxes.append((int(values[0]), int(values[1]), 
                                          int(values[2]), int(values[3])))
        
        elif ext == '.csv':
            import csv
            with open(bbox_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 4:
                        bboxes.append((int(float(row[0])), int(float(row[1])),
                                      int(float(row[2])), int(float(row[3]))))
    
    except Exception as e:
        print(f"Warning: Could not load bboxes from {bbox_path}: {e}")
    
    return bboxes


def convert_yolo_bboxes(bboxes: List, img_width: int, img_height: int) -> List[Tuple[int, int, int, int]]:
    """Convert YOLO normalized bboxes to absolute pixel coordinates."""
    converted = []
    for bbox in bboxes:
        if isinstance(bbox, tuple) and bbox[0] == 'yolo':
            _, cx, cy, w, h = bbox
            abs_cx = cx * img_width
            abs_cy = cy * img_height
            abs_w = w * img_width
            abs_h = h * img_height
            converted.append((
                int(abs_cx - abs_w/2),
                int(abs_cy - abs_h/2),
                int(abs_w),
                int(abs_h)
            ))
        else:
            converted.append(bbox)
    return converted


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_single_image(args: Tuple[str, str, Optional[CalibrationThresholds]]) -> Dict:
    """
    Analyze a single image. Used for multiprocessing.
    Returns raw metrics for calibration or full analysis for inference.
    """
    image_path, bbox_path, thresholds = args
    
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f"Could not load image: {image_path}", 'image_id': image_id}
        
        img_height, img_width = image.shape[:2]
        
        # Load bboxes
        bboxes = load_bboxes(bbox_path)
        bboxes = convert_yolo_bboxes(bboxes, img_width, img_height)
        
        # Compute image-level metrics
        metrics_computer = QualityMetricsComputer()
        image_metrics = metrics_computer.compute_all_metrics(image)
        
        # Compute pill-level metrics
        pill_analyzer = PillAnalyzer()
        pill_metrics_list = []
        
        if bboxes:
            areas = [w * h for x, y, w, h in bboxes if isinstance((x, y, w, h), tuple)]
            median_area = np.median(areas) if areas else 0
            
            for i, bbox in enumerate(bboxes):
                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    continue
                pill_metrics = pill_analyzer.analyze_pill(image, bbox, i, bboxes, median_area)
                pill_metrics_list.append(pill_metrics)
        
        result = {
            'image_id': image_id,
            'image_path': image_path,
            'image_metrics': asdict(image_metrics),
            'pill_metrics': [asdict(pm) for pm in pill_metrics_list],
            'total_pills': len(pill_metrics_list)
        }
        
        # If thresholds provided, perform classification
        if thresholds is not None:
            result.update(classify_image(result, thresholds))
        
        return result
    
    except Exception as e:
        import traceback
        return {
            'error': f"Error processing {image_path}: {str(e)}\n{traceback.format_exc()}",
            'image_id': image_id
        }


def classify_image(result: Dict, thresholds: CalibrationThresholds) -> Dict:
    """
    Classify an image based on calibrated thresholds.
    Returns classification results with reasons.
    """
    metrics = result['image_metrics']
    pill_metrics_list = result['pill_metrics']
    
    image_level_issues = []
    pill_level_issues = []
    
    # =========================================================================
    # IMAGE-LEVEL CLASSIFICATION
    # =========================================================================
    
    # Blur detection
    if metrics['laplacian_variance'] < thresholds.laplacian_variance_threshold:
        image_level_issues.append("global_blur_laplacian")
    
    if metrics['tenengrad'] < thresholds.tenengrad_threshold:
        image_level_issues.append("global_blur_tenengrad")
    
    if metrics['fft_high_freq_ratio'] < thresholds.fft_high_freq_threshold:
        image_level_issues.append("global_blur_fft")
    
    # Motion blur
    if metrics['motion_blur_score'] > thresholds.motion_blur_threshold:
        image_level_issues.append(f"motion_blur_detected_angle_{metrics['motion_blur_angle']:.0f}")
    
    # Brightness issues
    if metrics['mean_brightness'] < thresholds.brightness_low_threshold:
        image_level_issues.append("image_too_dim")
    
    if metrics['mean_brightness'] > thresholds.brightness_high_threshold:
        image_level_issues.append("image_too_bright")
    
    # Overexposure
    if metrics['overexposed_ratio'] > thresholds.overexposed_threshold:
        image_level_issues.append(f"overexposed_{metrics['overexposed_ratio']*100:.1f}%")
    
    # Underexposure
    if metrics['underexposed_ratio'] > thresholds.underexposed_threshold:
        image_level_issues.append(f"underexposed_{metrics['underexposed_ratio']*100:.1f}%")
    
    # Partial brightness issues
    if metrics['brightness_uniformity'] < thresholds.brightness_uniformity_threshold:
        image_level_issues.append("uneven_brightness")
    
    # Partial blur issues
    if metrics['blur_uniformity'] < thresholds.blur_uniformity_threshold:
        image_level_issues.append("partial_blur")
    
    # Light intrusion
    if metrics['light_blob_score'] > thresholds.light_blob_threshold:
        image_level_issues.append("external_light_intrusion")
    
    if metrics['saturation_peaks'] > thresholds.saturation_peaks_threshold:
        image_level_issues.append("high_saturation_anomaly")
    
    # Low contrast
    if metrics['rms_contrast'] < thresholds.contrast_threshold:
        image_level_issues.append("low_contrast")
    
    # =========================================================================
    # PILL-LEVEL CLASSIFICATION
    # =========================================================================
    
    bad_pill_count = 0
    stacking_count = 0
    blurry_pill_count = 0
    bright_pill_count = 0
    dim_pill_count = 0
    motion_blur_pill_count = 0
    
    for pm in pill_metrics_list:
        pill_issues = []
        
        # Stacking detection
        if pm['overlap_ratio'] > thresholds.pill_overlap_threshold:
            pill_issues.append("stacking")
            stacking_count += 1
        
        # Abnormal aspect ratio (might indicate partial pill or stacking)
        if (pm['aspect_ratio'] < 1/thresholds.pill_aspect_ratio_threshold or 
            pm['aspect_ratio'] > thresholds.pill_aspect_ratio_threshold):
            pill_issues.append("abnormal_aspect_ratio")
        
        # Size anomaly
        if pm['area_deviation'] > thresholds.pill_area_deviation_threshold:
            pill_issues.append("size_anomaly")
        
        # Pill blur
        if pm['laplacian_variance'] < thresholds.pill_blur_threshold:
            pill_issues.append("pill_blur")
            blurry_pill_count += 1
        
        # Pill brightness
        if pm['mean_brightness'] > thresholds.pill_brightness_high:
            pill_issues.append("pill_overexposed")
            bright_pill_count += 1
        
        if pm['mean_brightness'] < thresholds.pill_brightness_low:
            pill_issues.append("pill_underexposed")
            dim_pill_count += 1
        
        # Motion blur on pill
        if pm['motion_blur_score'] > thresholds.motion_blur_threshold:
            pill_issues.append("pill_motion_blur")
            motion_blur_pill_count += 1
        
        if pill_issues:
            bad_pill_count += 1
        
        pm['issues'] = pill_issues
    
    # Aggregate pill-level issues
    if stacking_count > 0:
        pill_level_issues.append(f"stacking_detected_{stacking_count}_pills")
    
    if blurry_pill_count > 0:
        pill_level_issues.append(f"blurry_pills_{blurry_pill_count}")
    
    if bright_pill_count > 0:
        pill_level_issues.append(f"overexposed_pills_{bright_pill_count}")
    
    if dim_pill_count > 0:
        pill_level_issues.append(f"underexposed_pills_{dim_pill_count}")
    
    if motion_blur_pill_count > 0:
        pill_level_issues.append(f"motion_blur_pills_{motion_blur_pill_count}")
    
    # =========================================================================
    # FINAL CLASSIFICATION
    # =========================================================================
    
    all_issues = image_level_issues + pill_level_issues
    is_bad = len(all_issues) > 0
    
    # Compute quality score (0-1, higher is better)
    quality_score = 1.0
    quality_score -= len(image_level_issues) * 0.15
    if pill_metrics_list:
        quality_score -= (bad_pill_count / len(pill_metrics_list)) * 0.3
    quality_score = max(0.0, quality_score)
    
    return {
        'is_bad': is_bad,
        'quality_score': quality_score,
        'image_level_issues': image_level_issues,
        'pill_level_issues': pill_level_issues,
        'bad_pill_count': bad_pill_count,
        'total_pill_count': len(pill_metrics_list)
    }


# =============================================================================
# CALIBRATION
# =============================================================================

def calibrate_thresholds(results: List[Dict], percentile: float = 15.0) -> CalibrationThresholds:
    """
    Analyze all results and determine thresholds to filter worst X%.
    """
    thresholds = CalibrationThresholds()
    
    # Collect all metrics
    laplacian_values = []
    tenengrad_values = []
    fft_values = []
    brightness_values = []
    overexposed_values = []
    underexposed_values = []
    brightness_uniformity_values = []
    blur_uniformity_values = []
    light_blob_values = []
    saturation_peaks_values = []
    motion_blur_values = []
    contrast_values = []
    
    pill_blur_values = []
    pill_brightness_values = []
    pill_overlap_values = []
    pill_aspect_ratios = []
    pill_area_deviations = []
    
    for r in results:
        if 'error' in r:
            continue
        
        m = r['image_metrics']
        laplacian_values.append(m['laplacian_variance'])
        tenengrad_values.append(m['tenengrad'])
        fft_values.append(m['fft_high_freq_ratio'])
        brightness_values.append(m['mean_brightness'])
        overexposed_values.append(m['overexposed_ratio'])
        underexposed_values.append(m['underexposed_ratio'])
        brightness_uniformity_values.append(m['brightness_uniformity'])
        blur_uniformity_values.append(m['blur_uniformity'])
        light_blob_values.append(m['light_blob_score'])
        saturation_peaks_values.append(m['saturation_peaks'])
        motion_blur_values.append(m['motion_blur_score'])
        contrast_values.append(m['rms_contrast'])
        
        for pm in r['pill_metrics']:
            pill_blur_values.append(pm['laplacian_variance'])
            pill_brightness_values.append(pm['mean_brightness'])
            pill_overlap_values.append(pm['overlap_ratio'])
            pill_aspect_ratios.append(pm['aspect_ratio'])
            pill_area_deviations.append(pm['area_deviation'])
    
    # Compute thresholds at specified percentile
    # For metrics where lower = worse (blur, contrast), use lower percentile
    # For metrics where higher = worse (overexposure, motion blur), use upper percentile
    
    if laplacian_values:
        thresholds.laplacian_variance_threshold = float(np.percentile(laplacian_values, percentile))
    
    if tenengrad_values:
        thresholds.tenengrad_threshold = float(np.percentile(tenengrad_values, percentile))
    
    if fft_values:
        thresholds.fft_high_freq_threshold = float(np.percentile(fft_values, percentile))
    
    if brightness_values:
        thresholds.brightness_low_threshold = float(np.percentile(brightness_values, percentile))
        thresholds.brightness_high_threshold = float(np.percentile(brightness_values, 100 - percentile))
    
    if overexposed_values:
        thresholds.overexposed_threshold = float(np.percentile(overexposed_values, 100 - percentile))
    
    if underexposed_values:
        thresholds.underexposed_threshold = float(np.percentile(underexposed_values, 100 - percentile))
    
    if brightness_uniformity_values:
        thresholds.brightness_uniformity_threshold = float(np.percentile(brightness_uniformity_values, percentile))
    
    if blur_uniformity_values:
        thresholds.blur_uniformity_threshold = float(np.percentile(blur_uniformity_values, percentile))
    
    if light_blob_values:
        thresholds.light_blob_threshold = float(np.percentile(light_blob_values, 100 - percentile))
    
    if saturation_peaks_values:
        thresholds.saturation_peaks_threshold = float(np.percentile(saturation_peaks_values, 100 - percentile))
    
    if motion_blur_values:
        thresholds.motion_blur_threshold = float(np.percentile(motion_blur_values, 100 - percentile))
    
    if contrast_values:
        thresholds.contrast_threshold = float(np.percentile(contrast_values, percentile))
    
    # Pill-level thresholds
    if pill_blur_values:
        thresholds.pill_blur_threshold = float(np.percentile(pill_blur_values, percentile))
    
    if pill_brightness_values:
        thresholds.pill_brightness_low = float(np.percentile(pill_brightness_values, percentile))
        thresholds.pill_brightness_high = float(np.percentile(pill_brightness_values, 100 - percentile))
    
    if pill_overlap_values:
        # For overlap, use a fixed percentile since most should be 0
        non_zero_overlaps = [o for o in pill_overlap_values if o > 0]
        if non_zero_overlaps:
            thresholds.pill_overlap_threshold = float(np.percentile(non_zero_overlaps, 50))
        else:
            thresholds.pill_overlap_threshold = 0.1  # Default
    
    if pill_aspect_ratios:
        # Aspect ratio: extreme values are bad
        thresholds.pill_aspect_ratio_threshold = float(np.percentile(pill_aspect_ratios, 100 - percentile/2))
    
    if pill_area_deviations:
        thresholds.pill_area_deviation_threshold = float(np.percentile(pill_area_deviations, 100 - percentile))
    
    # Store calibration statistics
    thresholds.calibration_stats = {
        'total_images': len([r for r in results if 'error' not in r]),
        'total_pills': len(pill_blur_values),
        'percentile_used': percentile,
        'laplacian_mean': float(np.mean(laplacian_values)) if laplacian_values else 0,
        'laplacian_std': float(np.std(laplacian_values)) if laplacian_values else 0,
        'brightness_mean': float(np.mean(brightness_values)) if brightness_values else 0,
        'brightness_std': float(np.std(brightness_values)) if brightness_values else 0,
    }
    
    return thresholds


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def find_image_bbox_pairs(images_dir: str, bboxes_dir: str) -> List[Tuple[str, str]]:
    """
    Match images with their corresponding bbox files.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    bbox_extensions = {'.json', '.txt', '.csv'}
    
    pairs = []
    
    # Get all images
    image_files = {}
    for f in os.listdir(images_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in image_extensions:
            image_files[name] = os.path.join(images_dir, f)
    
    # Get all bbox files
    bbox_files = {}
    for f in os.listdir(bboxes_dir):
        name, ext = os.path.splitext(f)
        if ext.lower() in bbox_extensions:
            bbox_files[name] = os.path.join(bboxes_dir, f)
    
    # Match pairs
    for name, image_path in image_files.items():
        bbox_path = bbox_files.get(name, '')
        pairs.append((image_path, bbox_path))
    
    return pairs


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================

class PillQualityAssessor:
    """Main class orchestrating the quality assessment pipeline."""
    
    def __init__(self, images_dir: str, bboxes_dir: str, output_dir: str,
                 num_workers: int = None, filter_percentile: float = 15.0):
        self.images_dir = images_dir
        self.bboxes_dir = bboxes_dir
        self.output_dir = output_dir
        self.num_workers = num_workers or max(1, mp.cpu_count() - 2)
        self.filter_percentile = filter_percentile
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.thresholds_path = os.path.join(output_dir, 'calibration_thresholds.pkl')
        self.results_path = os.path.join(output_dir, 'quality_assessment_results.json')
        self.metrics_path = os.path.join(output_dir, 'raw_metrics.pkl')
    
    def run_calibration(self) -> CalibrationThresholds:
        """
        Phase 1: Calibration - analyze all images and determine thresholds.
        """
        print("=" * 70)
        print("PHASE 1: CALIBRATION")
        print("=" * 70)
        
        # Find all image-bbox pairs
        pairs = find_image_bbox_pairs(self.images_dir, self.bboxes_dir)
        print(f"Found {len(pairs)} images to analyze")
        
        if not pairs:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # Prepare arguments for multiprocessing
        args_list = [(img, bbox, None) for img, bbox in pairs]
        
        # Run parallel analysis
        print(f"Analyzing images using {self.num_workers} workers...")
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(analyze_single_image, args): args[0] 
                      for args in args_list}
            
            with tqdm(total=len(futures), desc="Calibration") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if 'error' in result:
                        tqdm.write(f"Warning: {result['error'][:100]}")
        
        # Save raw metrics
        with open(self.metrics_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Raw metrics saved to: {self.metrics_path}")
        
        # Compute thresholds
        print(f"\nComputing thresholds at {self.filter_percentile}th percentile...")
        thresholds = calibrate_thresholds(results, self.filter_percentile)
        
        # Save thresholds
        with open(self.thresholds_path, 'wb') as f:
            pickle.dump(thresholds, f)
        print(f"Thresholds saved to: {self.thresholds_path}")
        
        # Print threshold summary
        self._print_threshold_summary(thresholds)
        
        return thresholds
    
    def run_inference(self, thresholds: CalibrationThresholds = None) -> List[Dict]:
        """
        Phase 2: Inference - classify all images using calibrated thresholds.
        """
        print("\n" + "=" * 70)
        print("PHASE 2: INFERENCE")
        print("=" * 70)
        
        # Load thresholds if not provided
        if thresholds is None:
            if not os.path.exists(self.thresholds_path):
                raise ValueError("No thresholds found. Run calibration first.")
            with open(self.thresholds_path, 'rb') as f:
                thresholds = pickle.load(f)
        
        # Check if we have raw metrics to reuse
        if os.path.exists(self.metrics_path):
            print("Loading cached metrics...")
            with open(self.metrics_path, 'rb') as f:
                raw_results = pickle.load(f)
            
            # Apply classification to cached results
            print("Applying classification...")
            final_results = []
            for r in tqdm(raw_results, desc="Classifying"):
                if 'error' not in r:
                    r.update(classify_image(r, thresholds))
                final_results.append(r)
        else:
            # Run full analysis with thresholds
            pairs = find_image_bbox_pairs(self.images_dir, self.bboxes_dir)
            args_list = [(img, bbox, thresholds) for img, bbox in pairs]
            
            final_results = []
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(analyze_single_image, args): args[0] 
                          for args in args_list}
                
                with tqdm(total=len(futures), desc="Inference") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        final_results.append(result)
                        pbar.update(1)
        
        # Generate output
        output = self._format_output(final_results)
        
        # Save results
        with open(self.results_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {self.results_path}")
        
        # Print summary
        self._print_summary(output)
        
        return output
    
    def run_full_pipeline(self) -> List[Dict]:
        """
        Run complete calibration + inference pipeline.
        """
        thresholds = self.run_calibration()
        return self.run_inference(thresholds)
    
    def _format_output(self, results: List[Dict]) -> List[Dict]:
        """
        Format results into the requested output structure.
        """
        output = []
        
        for r in results:
            if 'error' in r:
                output.append({
                    'image_id': r.get('image_id', 'unknown'),
                    'error': r['error']
                })
                continue
            
            entry = {
                'image_id': r['image_id'],
                'image_path': r['image_path'],
                'is_bad': r.get('is_bad', False),
                'quality_score': round(r.get('quality_score', 1.0), 4),
                'image_level_issues': r.get('image_level_issues', []),
                'pill_level_issues': r.get('pill_level_issues', []),
                'bad_pill_count': r.get('bad_pill_count', 0),
                'total_pill_count': r.get('total_pill_count', 0),
                'detailed_metrics': {
                    'laplacian_variance': round(r['image_metrics']['laplacian_variance'], 2),
                    'mean_brightness': round(r['image_metrics']['mean_brightness'], 2),
                    'motion_blur_score': round(r['image_metrics']['motion_blur_score'], 4),
                    'brightness_uniformity': round(r['image_metrics']['brightness_uniformity'], 4),
                    'blur_uniformity': round(r['image_metrics']['blur_uniformity'], 4),
                }
            }
            output.append(entry)
        
        # Sort by quality score (worst first)
        output.sort(key=lambda x: x.get('quality_score', 1.0))
        
        return output
    
    def _print_threshold_summary(self, thresholds: CalibrationThresholds):
        """Print calibration threshold summary."""
        print("\n" + "-" * 50)
        print("CALIBRATION THRESHOLDS (worst 15% filtering)")
        print("-" * 50)
        print(f"Images analyzed: {thresholds.calibration_stats.get('total_images', 'N/A')}")
        print(f"Pills analyzed: {thresholds.calibration_stats.get('total_pills', 'N/A')}")
        print()
        print("Image-Level Thresholds:")
        print(f"  Laplacian variance (blur):     < {thresholds.laplacian_variance_threshold:.2f}")
        print(f"  Tenengrad (blur):              < {thresholds.tenengrad_threshold:.2f}")
        print(f"  FFT high-freq ratio (blur):    < {thresholds.fft_high_freq_threshold:.4f}")
        print(f"  Motion blur score:             > {thresholds.motion_blur_threshold:.4f}")
        print(f"  Brightness low:                < {thresholds.brightness_low_threshold:.2f}")
        print(f"  Brightness high:               > {thresholds.brightness_high_threshold:.2f}")
        print(f"  Overexposed ratio:             > {thresholds.overexposed_threshold:.4f}")
        print(f"  Underexposed ratio:            > {thresholds.underexposed_threshold:.4f}")
        print(f"  Brightness uniformity:         < {thresholds.brightness_uniformity_threshold:.4f}")
        print(f"  Blur uniformity:               < {thresholds.blur_uniformity_threshold:.4f}")
        print(f"  Light blob score:              > {thresholds.light_blob_threshold:.6f}")
        print(f"  Contrast (RMS):                < {thresholds.contrast_threshold:.4f}")
        print()
        print("Pill-Level Thresholds:")
        print(f"  Pill blur (Laplacian):         < {thresholds.pill_blur_threshold:.2f}")
        print(f"  Pill brightness low:           < {thresholds.pill_brightness_low:.2f}")
        print(f"  Pill brightness high:          > {thresholds.pill_brightness_high:.2f}")
        print(f"  Overlap ratio (stacking):      > {thresholds.pill_overlap_threshold:.4f}")
        print(f"  Aspect ratio extreme:          > {thresholds.pill_aspect_ratio_threshold:.4f}")
        print(f"  Area deviation:                > {thresholds.pill_area_deviation_threshold:.4f}")
        print("-" * 50)
    
    def _print_summary(self, results: List[Dict]):
        """Print inference summary."""
        total = len(results)
        errors = sum(1 for r in results if 'error' in r)
        bad = sum(1 for r in results if r.get('is_bad', False))
        good = total - errors - bad
        
        print("\n" + "=" * 50)
        print("QUALITY ASSESSMENT SUMMARY")
        print("=" * 50)
        print(f"Total images:    {total}")
        print(f"Good images:     {good} ({100*good/total:.1f}%)")
        print(f"Bad images:      {bad} ({100*bad/total:.1f}%)")
        print(f"Errors:          {errors}")
        print()
        
        # Issue frequency analysis
        issue_counts = {}
        for r in results:
            for issue in r.get('image_level_issues', []):
                # Normalize issue names for counting
                base_issue = issue.split('_')[0] if issue.startswith(('overexposed', 'underexposed')) else issue
                issue_counts[base_issue] = issue_counts.get(base_issue, 0) + 1
            for issue in r.get('pill_level_issues', []):
                base_issue = issue.split('_')[0]
                issue_counts[base_issue] = issue_counts.get(base_issue, 0) + 1
        
        if issue_counts:
            print("Top Issues Detected:")
            for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {issue}: {count} images")
        
        print("=" * 50)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pill Image Quality Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (calibration + inference)
  python pill_quality_assessment.py --images ./images_consolidated --bboxes ./bboxes --output ./results

  # Run only calibration
  python pill_quality_assessment.py --images ./images_consolidated --bboxes ./bboxes --output ./results --calibrate-only

  # Run inference with existing thresholds
  python pill_quality_assessment.py --images ./images_consolidated --bboxes ./bboxes --output ./results --inference-only

  # Custom percentile for filtering
  python pill_quality_assessment.py --images ./images_consolidated --bboxes ./bboxes --output ./results --percentile 20
        """
    )
    
    parser.add_argument('--images', '-i', required=True,
                       help='Path to images directory')
    parser.add_argument('--bboxes', '-b', required=True,
                       help='Path to bounding boxes directory')
    parser.add_argument('--output', '-o', required=True,
                       help='Path to output directory')
    parser.add_argument('--percentile', '-p', type=float, default=15.0,
                       help='Percentile for filtering worst images (default: 15)')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 2)')
    parser.add_argument('--calibrate-only', action='store_true',
                       help='Only run calibration phase')
    parser.add_argument('--inference-only', action='store_true',
                       help='Only run inference phase (requires existing thresholds)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.images):
        print(f"Error: Images directory not found: {args.images}")
        return 1
    
    if not os.path.isdir(args.bboxes):
        print(f"Error: Bboxes directory not found: {args.bboxes}")
        return 1
    
    # Create assessor
    assessor = PillQualityAssessor(
        images_dir=args.images,
        bboxes_dir=args.bboxes,
        output_dir=args.output,
        num_workers=args.workers,
        filter_percentile=args.percentile
    )
    
    # Run appropriate phase
    try:
        if args.calibrate_only:
            assessor.run_calibration()
        elif args.inference_only:
            assessor.run_inference()
        else:
            assessor.run_full_pipeline()
        
        print("\nDone!")
        return 0
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
