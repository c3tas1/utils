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

# Force CPU-only mode - prevent any library from using GPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENCV_OPENCL_DEVICE'] = 'disabled'
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
    light_in_pill_region: float = 0.0  # Light affecting pills (bad)
    light_in_background: float = 0.0   # Light in background only (acceptable)
    
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
    is_stacking: bool = False  # True only for actual vertical stacking
    aspect_ratio: float = 0.0
    
    # Pill orientation detection (side vs flat)
    is_on_side: bool = False  # True if pill is on its edge instead of flat
    orientation_score: float = 0.0  # Higher = more likely on side
    edge_thickness_ratio: float = 0.0  # Thin profile suggests side placement
    area_deviation: float = 0.0  # deviation from median pill area
    
    # Edge quality
    edge_density: float = 0.0
    edge_continuity: float = 0.0
    
    # Motion blur (pill still settling/moving when captured)
    motion_blur_score: float = 0.0
    has_motion_blur: bool = False
    
    # Same-image outlier detection flags
    is_blur_outlier: bool = False  # Blurrier than siblings in same image
    is_brightness_outlier: bool = False  # Brighter/dimmer than siblings
    brightness_outlier_type: str = ""  # "overexposed" or "underexposed"
    is_motion_outlier: bool = False  # More motion blur than siblings
    
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
        Detect global motion blur using directional FFT analysis.
        Note: For pill-specific motion blur (settling pills), use compute_pill_motion_blur instead.
        Returns (motion_blur_score, dominant_angle).
        """
        # Simplified global motion blur - less aggressive
        rows, cols = gray.shape
        
        # Compute gradient magnitudes in different directions
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # If blur is directional, gradients will be weak in blur direction
        gx_energy = np.mean(np.abs(gx))
        gy_energy = np.mean(np.abs(gy))
        
        # Anisotropy in gradients suggests directional blur
        max_energy = max(gx_energy, gy_energy)
        min_energy = min(gx_energy, gy_energy)
        
        if max_energy > 0:
            anisotropy = (max_energy - min_energy) / (max_energy + 1e-10)
        else:
            anisotropy = 0.0
        
        # Determine dominant angle (0 = horizontal blur, 90 = vertical blur)
        if gx_energy < gy_energy:
            dominant_angle = 0.0  # Horizontal motion blur (weak horizontal edges)
        else:
            dominant_angle = 90.0  # Vertical motion blur
        
        return float(anisotropy), float(dominant_angle)
    
    @staticmethod
    def compute_pill_motion_blur(gray: np.ndarray) -> Dict[str, float]:
        """
        Detect motion blur on individual pill crops.
        
        Motion blur from settling pills causes:
        1. Directional smearing/streaking
        2. Elongated edges in motion direction
        3. Reduced sharpness with directional bias
        4. Ghosting/double edges
        
        Returns dict with motion blur indicators.
        """
        if gray.shape[0] < 10 or gray.shape[1] < 10:
            return {'motion_score': 0.0, 'has_motion_blur': False, 'direction': 0.0}
        
        h, w = gray.shape
        
        # 1. Directional gradient analysis
        # Motion blur causes weak gradients in motion direction, strong perpendicular
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        grad_x_energy = np.mean(np.abs(sobel_x))
        grad_y_energy = np.mean(np.abs(sobel_y))
        
        # Gradient anisotropy - high value means directional blur
        total_grad = grad_x_energy + grad_y_energy + 1e-10
        grad_anisotropy = abs(grad_x_energy - grad_y_energy) / total_grad
        
        # 2. Edge spread analysis using Laplacian
        # Motion blur spreads edges, reducing Laplacian response
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # 3. Streak detection using morphological operations
        # Motion blur creates elongated structures
        edges = cv2.Canny(gray, 50, 150)
        
        # Check for elongated edge structures (streaks)
        # Use line-shaped kernels to detect directional patterns
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        
        # Opening with directional kernels - preserves streaks in that direction
        streaks_h = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
        streaks_v = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v)
        
        streak_h_ratio = np.sum(streaks_h > 0) / (edges.size + 1e-10)
        streak_v_ratio = np.sum(streaks_v > 0) / (edges.size + 1e-10)
        streak_anisotropy = abs(streak_h_ratio - streak_v_ratio) / (streak_h_ratio + streak_v_ratio + 1e-10)
        
        # 4. Double edge / ghosting detection
        # Motion blur can create parallel edge responses
        # Use dilated edge comparison
        kernel_small = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel_small, iterations=2)
        edge_spread = np.sum(edges_dilated > 0) / (np.sum(edges > 0) + 1e-10)
        
        # 5. Frequency domain check - motion blur attenuates high frequencies directionally
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))
        
        # Check energy distribution in horizontal vs vertical bands
        center_y, center_x = h // 2, w // 2
        band_width = min(h, w) // 8
        
        # Horizontal band (detects vertical motion blur)
        h_band = magnitude[center_y - 2:center_y + 2, :]
        # Vertical band (detects horizontal motion blur)  
        v_band = magnitude[:, center_x - 2:center_x + 2]
        
        h_band_energy = np.mean(h_band) if h_band.size > 0 else 0
        v_band_energy = np.mean(v_band) if v_band.size > 0 else 0
        
        fft_anisotropy = abs(h_band_energy - v_band_energy) / (h_band_energy + v_band_energy + 1e-10)
        
        # Combine metrics into motion blur score
        # Weight factors tuned for pill motion blur detection
        motion_score = (
            0.3 * grad_anisotropy +      # Directional gradient weakness
            0.25 * streak_anisotropy +    # Elongated edge structures
            0.25 * fft_anisotropy +       # Frequency domain anisotropy
            0.2 * min(edge_spread / 5.0, 1.0)  # Edge spreading/ghosting
        )
        
        # Determine motion direction
        if grad_x_energy < grad_y_energy:
            direction = 0.0  # Horizontal motion (blur in X direction)
        else:
            direction = 90.0  # Vertical motion (blur in Y direction)
        
        # Motion blur threshold - calibrated for pill settling motion
        # Score > 0.35 suggests significant motion blur
        has_motion_blur = motion_score > 0.35
        
        return {
            'motion_score': float(motion_score),
            'has_motion_blur': has_motion_blur,
            'direction': direction,
            'grad_anisotropy': float(grad_anisotropy),
            'streak_anisotropy': float(streak_anisotropy),
            'fft_anisotropy': float(fft_anisotropy)
        }
    
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
    def compute_light_intrusion_metrics(image: np.ndarray, 
                                        bboxes: List[Tuple[int, int, int, int]] = None) -> Dict[str, float]:
        """
        Detect external light intrusion that affects PILL REGIONS.
        
        Light in background (non-pill areas) is acceptable.
        Only flag light intrusion if it's in/near pill bounding boxes.
        
        Args:
            image: Full image (BGR or grayscale)
            bboxes: List of pill bounding boxes (x, y, w, h). If provided,
                   only analyzes light intrusion in pill regions.
        
        Returns:
            Dict with light intrusion metrics
        """
        # Convert to HSV
        if len(image.shape) == 2:
            gray = image
            hsv = None
        else:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        img_h, img_w = gray.shape[:2]
        
        results = {
            'saturation_peaks': 0.0,
            'high_saturation_ratio': 0.0,
            'light_blob_score': 0.0,
            'light_in_pill_region': 0.0,  # NEW: light specifically affecting pills
            'light_in_background': 0.0,   # NEW: light in background (acceptable)
        }
        
        # Create pill region mask if bboxes provided
        pill_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if bboxes:
            for bbox in bboxes:
                if isinstance(bbox, tuple) and len(bbox) == 4:
                    x, y, w, h = bbox
                    # Expand bbox slightly to catch light bleeding onto pills
                    expand = 5
                    x1 = max(0, int(x) - expand)
                    y1 = max(0, int(y) - expand)
                    x2 = min(img_w, int(x + w) + expand)
                    y2 = min(img_h, int(y + h) + expand)
                    pill_mask[y1:y2, x1:x2] = 255
        
        background_mask = cv2.bitwise_not(pill_mask) if bboxes else None
        
        # Detect bright blobs (potential light intrusion)
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        
        # Find contours of bright regions
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_light_area = 0
        light_in_pill_area = 0
        light_in_background_area = 0
        
        if contours:
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area < 50:  # Skip tiny bright spots (noise)
                    continue
                
                total_light_area += contour_area
                
                if bboxes and pill_mask is not None:
                    # Create mask for this contour
                    contour_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                    
                    # Check overlap with pill regions
                    overlap_with_pills = cv2.bitwise_and(contour_mask, pill_mask)
                    overlap_area = np.sum(overlap_with_pills > 0)
                    
                    light_in_pill_area += overlap_area
                    light_in_background_area += (contour_area - overlap_area)
        
        # Calculate ratios
        if bboxes:
            pill_region_area = np.sum(pill_mask > 0)
            background_area = img_h * img_w - pill_region_area
            
            # Light in pill region ratio (THIS IS THE PROBLEM)
            if pill_region_area > 0:
                results['light_in_pill_region'] = float(light_in_pill_area / pill_region_area)
            
            # Light in background ratio (this is OK)
            if background_area > 0:
                results['light_in_background'] = float(light_in_background_area / background_area)
            
            # Overall light blob score - ONLY count light affecting pills
            results['light_blob_score'] = results['light_in_pill_region']
        else:
            # No bboxes provided - use total light area (old behavior)
            results['light_blob_score'] = float(total_light_area / gray.size)
        
        # Saturation analysis (for color images)
        if hsv is not None:
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # High saturation with high value often indicates external light
            high_sat_high_val = (saturation > 150) & (value > 200)
            
            if bboxes and pill_mask is not None:
                # Only count high saturation IN pill regions
                high_sat_in_pills = high_sat_high_val & (pill_mask > 0)
                pill_pixels = np.sum(pill_mask > 0)
                if pill_pixels > 0:
                    results['high_saturation_ratio'] = float(np.sum(high_sat_in_pills) / pill_pixels)
            else:
                results['high_saturation_ratio'] = float(np.sum(high_sat_high_val) / saturation.size)
            
            # Peak saturation in pill regions only
            if bboxes and pill_mask is not None:
                pill_saturation = saturation[pill_mask > 0]
                if pill_saturation.size > 0:
                    results['saturation_peaks'] = float(np.percentile(pill_saturation, 99) / 255.0)
            else:
                results['saturation_peaks'] = float(np.percentile(saturation, 99) / 255.0)
        
        return results
    
    def compute_all_metrics(self, image: np.ndarray, 
                            bboxes: List[Tuple[int, int, int, int]] = None) -> QualityMetrics:
        """
        Compute all quality metrics for an image.
        
        Args:
            image: Input image (BGR or grayscale)
            bboxes: Optional list of pill bounding boxes for targeted analysis
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
        
        # Light intrusion - pass bboxes to only flag light affecting pills
        light = self.compute_light_intrusion_metrics(image, bboxes)
        metrics.saturation_peaks = light['saturation_peaks']
        metrics.high_saturation_ratio = light['high_saturation_ratio']
        metrics.light_blob_score = light['light_blob_score']
        metrics.light_in_pill_region = light.get('light_in_pill_region', 0.0)
        metrics.light_in_background = light.get('light_in_background', 0.0)
        
        return metrics


# =============================================================================
# PILL-LEVEL ANALYSIS
# =============================================================================

class PillAnalyzer:
    """Analyze individual pills from bounding boxes."""
    
    def __init__(self):
        self.metrics_computer = QualityMetricsComputer()
    
    
    def compute_overlap_ratio(self, bbox: Tuple[int, int, int, int], 
                             all_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[float, bool]:
        """
        Detect TRUE stacking - one pill physically on top of another.
        
        Stacking criteria:
        - Significant intersection area (>20% of the smaller pill)
        - Not just edge touching
        
        Returns:
            (max_overlap_ratio, is_stacking)
        """
        x1, y1, w1, h1 = bbox
        box1_area = w1 * h1
        
        if box1_area == 0:
            return 0.0, False
        
        max_overlap = 0.0
        is_stacking = False
        
        for other in all_bboxes:
            if other == bbox:
                continue
            
            x2, y2, w2, h2 = other
            box2_area = w2 * h2
            
            if box2_area == 0:
                continue
            
            # Compute intersection
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix2 > ix1 and iy2 > iy1:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                
                # Overlap relative to smaller box
                smaller_area = min(box1_area, box2_area)
                overlap_ratio = intersection_area / smaller_area
                
                if overlap_ratio > max_overlap:
                    max_overlap = overlap_ratio
                
                # Stacking = significant overlap (>20% of smaller pill)
                if overlap_ratio > 0.20:
                    is_stacking = True
        
        return max_overlap, is_stacking
    
    def detect_pill_orientation(self, bbox: Tuple[int, int, int, int],
                                bbox_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect if pill is placed on its side using simple, robust metrics.
        
        Side-placed pill characteristics:
        1. Significantly more elongated than siblings (high aspect ratio)
        2. Significantly smaller area than siblings
        
        Uses IQR-based outlier detection for robustness.
        """
        x, y, w, h = bbox
        area = w * h
        aspect_ratio = max(w/h, h/w) if h > 0 and w > 0 else 1.0  # Always >= 1
        
        # Get statistics
        median_area = bbox_stats.get('median_area', area)
        q1_area = bbox_stats.get('q1_area', median_area * 0.7)
        q3_area = bbox_stats.get('q3_area', median_area * 1.3)
        
        median_aspect = bbox_stats.get('median_aspect_ratio', 1.0)
        q1_aspect = bbox_stats.get('q1_aspect', 1.0)
        q3_aspect = bbox_stats.get('q3_aspect', 1.5)
        
        # IQR calculations
        iqr_area = q3_area - q1_area
        iqr_aspect = q3_aspect - q1_aspect
        
        # Outlier thresholds (1.5 * IQR is standard)
        area_lower_fence = q1_area - 1.5 * iqr_area
        aspect_upper_fence = q3_aspect + 1.5 * iqr_aspect
        
        # Detection flags
        is_small_outlier = area < area_lower_fence if iqr_area > 100 else False
        is_elongated_outlier = aspect_ratio > aspect_upper_fence if iqr_aspect > 0.1 else False
        
        # Side placement = elongated AND/OR unusually small
        # Being both is strong evidence, being one is weak evidence
        is_on_side = False
        confidence = 0.0
        
        if is_elongated_outlier and is_small_outlier:
            # Strong evidence - both criteria met
            is_on_side = True
            confidence = 0.9
        elif is_elongated_outlier and (area < median_area * 0.7):
            # Elongated and somewhat small
            is_on_side = True
            confidence = 0.7
        elif aspect_ratio > 2.5 and area < median_area * 0.5:
            # Very elongated and quite small (absolute thresholds)
            is_on_side = True
            confidence = 0.6
        
        return {
            'is_on_side': is_on_side,
            'confidence': confidence,
            'aspect_ratio': aspect_ratio,
            'area_ratio': area / median_area if median_area > 0 else 1.0,
            'is_small_outlier': is_small_outlier,
            'is_elongated_outlier': is_elongated_outlier
        }
    
    def analyze_pill(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                    bbox_id: int, all_bboxes: List[Tuple[int, int, int, int]],
                    bbox_stats: Dict[str, Any]) -> PillMetrics:
        """
        Analyze a single pill crop.
        
        Args:
            image: Full image
            bbox: This pill's bounding box (x, y, w, h)
            bbox_id: Index of this bbox
            all_bboxes: All bboxes in this image (for overlap detection)
            bbox_stats: Statistics from all bboxes for outlier detection
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
        
        # Stacking detection - now returns (overlap_ratio, is_true_stacking)
        overlap_ratio, is_stacking = self.compute_overlap_ratio(bbox, all_bboxes)
        metrics.overlap_ratio = overlap_ratio
        metrics.is_stacking = is_stacking
        
        # Aspect ratio and area deviation
        metrics.aspect_ratio = w / h if h > 0 else 0
        area = w * h
        median_area = bbox_stats.get('median_area', area)
        if median_area > 0:
            metrics.area_deviation = abs(area - median_area) / median_area
        
        # Pill orientation detection (on side vs flat) - simple geometric comparison
        orientation = self.detect_pill_orientation(bbox, bbox_stats)
        metrics.is_on_side = orientation['is_on_side']
        metrics.orientation_score = orientation['confidence']
        metrics.edge_thickness_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0
        
        # Edge analysis for partial occlusion detection
        edges = cv2.Canny(gray, 50, 150)
        metrics.edge_density = float(np.sum(edges > 0) / edges.size)
        
        # Edge continuity - check if edges form complete contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            metrics.edge_continuity = contour_area / (w * h) if w * h > 0 else 0
        
        # Motion blur for this pill - detect settling motion
        if gray.shape[0] > 10 and gray.shape[1] > 10:
            motion_result = self.metrics_computer.compute_pill_motion_blur(gray)
            metrics.motion_blur_score = motion_result['motion_score']
            metrics.has_motion_blur = motion_result['has_motion_blur']
        else:
            metrics.motion_blur_score = 0.0
            metrics.has_motion_blur = False
        
        # === SAME-IMAGE OUTLIER DETECTION ===
        # Compare this pill against its siblings in the same image
        # A pill with issues will be an outlier compared to the majority
        
        # Blur outlier detection (this pill blurrier than siblings)
        median_lap = bbox_stats.get('median_laplacian', metrics.laplacian_variance)
        lap_std = bbox_stats.get('laplacian_std', 0)
        q1_lap = bbox_stats.get('q1_laplacian', median_lap)
        
        if lap_std > 10 and median_lap > 0:  # Only if there's meaningful variation
            # IQR method
            iqr_lap = bbox_stats.get('q3_laplacian', median_lap) - q1_lap
            blur_lower_fence = q1_lap - 1.5 * iqr_lap
            
            # Z-score method
            lap_zscore = (median_lap - metrics.laplacian_variance) / (lap_std + 1e-10)
            
            # Flag if significantly blurrier than siblings
            if metrics.laplacian_variance < blur_lower_fence or lap_zscore > 2.0:
                metrics.is_blur_outlier = True
        
        # Brightness outlier detection (this pill over/under exposed vs siblings)
        median_bright = bbox_stats.get('median_brightness', metrics.mean_brightness)
        bright_std = bbox_stats.get('brightness_std', 0)
        q1_bright = bbox_stats.get('q1_brightness', median_bright * 0.9)
        q3_bright = bbox_stats.get('q3_brightness', median_bright * 1.1)
        
        if bright_std > 5 and median_bright > 0:  # Only if there's meaningful variation
            iqr_bright = q3_bright - q1_bright
            bright_lower_fence = q1_bright - 1.5 * iqr_bright
            bright_upper_fence = q3_bright + 1.5 * iqr_bright
            
            # Z-score
            bright_zscore = abs(metrics.mean_brightness - median_bright) / (bright_std + 1e-10)
            
            if metrics.mean_brightness < bright_lower_fence or \
               (bright_zscore > 2.0 and metrics.mean_brightness < median_bright):
                metrics.is_brightness_outlier = True
                metrics.brightness_outlier_type = "underexposed"
            elif metrics.mean_brightness > bright_upper_fence or \
                 (bright_zscore > 2.0 and metrics.mean_brightness > median_bright):
                metrics.is_brightness_outlier = True
                metrics.brightness_outlier_type = "overexposed"
        
        # Motion blur outlier detection (this pill has motion blur while siblings don't)
        median_motion = bbox_stats.get('median_motion', 0)
        motion_std = bbox_stats.get('motion_std', 0)
        q3_motion = bbox_stats.get('q3_motion', median_motion)
        
        if motion_std > 0.05:  # Only if there's meaningful variation
            iqr_motion = q3_motion - bbox_stats.get('q1_motion', median_motion)
            motion_upper_fence = q3_motion + 1.5 * iqr_motion
            
            # Z-score
            motion_zscore = (metrics.motion_blur_score - median_motion) / (motion_std + 1e-10)
            
            # Flag if significantly more motion blur than siblings
            if metrics.motion_blur_score > motion_upper_fence or motion_zscore > 2.0:
                metrics.is_motion_outlier = True
                # Also set has_motion_blur if detected as outlier
                if metrics.motion_blur_score > 0.2:  # Lower threshold for outlier-based detection
                    metrics.has_motion_blur = True
        
        return metrics


# =============================================================================
# BBOX LOADING
# =============================================================================

def load_bboxes(bbox_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Load bounding boxes from various formats.
    Supports: JSON, YOLO format txt, CSV, Pickle
    Returns list of (x, y, w, h) tuples in pixel coordinates.
    """
    bboxes = []
    
    if not os.path.exists(bbox_path):
        return bboxes
    
    ext = os.path.splitext(bbox_path)[1].lower()
    
    try:
        if ext == '.pkl' or ext == '.pickle':
            # Pickle format: [[x, y, w, h], ...] OR [[x1, y1, x2, y2], ...]
            with open(bbox_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, (list, np.ndarray)):
                for item in data:
                    if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 4:
                        # Handle numpy floats/ints
                        v0, v1, v2, v3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                        
                        # Auto-detect format: x1,y1,x2,y2 vs x,y,w,h
                        # If v2 > v0 and v3 > v1 significantly, likely x1,y1,x2,y2 format
                        # (x2 should be > x1 and y2 should be > y1)
                        # For x,y,w,h format, w and h are typically much smaller than x,y
                        
                        # Heuristic: if v2 > v0 and v3 > v1, and the "width" (v2-v0) is reasonable
                        # compared to v2 and v3, it's probably x1,y1,x2,y2
                        if v2 > v0 and v3 > v1:
                            # Likely x1, y1, x2, y2 format - convert to x, y, w, h
                            x, y = int(v0), int(v1)
                            w, h = int(v2 - v0), int(v3 - v1)
                        else:
                            # Assume x, y, w, h format
                            x, y, w, h = int(v0), int(v1), int(v2), int(v3)
                        
                        if w > 0 and h > 0:
                            bboxes.append((x, y, w, h))
                            
            elif isinstance(data, dict):
                # In case it's a dict with 'bboxes' key or similar
                for key in ['bboxes', 'boxes', 'detections', 'bbox']:
                    if key in data:
                        for item in data[key]:
                            if len(item) >= 4:
                                v0, v1, v2, v3 = float(item[0]), float(item[1]), float(item[2]), float(item[3])
                                if v2 > v0 and v3 > v1:
                                    x, y = int(v0), int(v1)
                                    w, h = int(v2 - v0), int(v3 - v1)
                                else:
                                    x, y, w, h = int(v0), int(v1), int(v2), int(v3)
                                if w > 0 and h > 0:
                                    bboxes.append((x, y, w, h))
                        break
        
        elif ext == '.json':
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
        
        # Compute image-level metrics (pass bboxes for targeted light intrusion detection)
        metrics_computer = QualityMetricsComputer()
        image_metrics = metrics_computer.compute_all_metrics(image, bboxes)
        
        # Compute pill-level metrics
        pill_analyzer = PillAnalyzer()
        pill_metrics_list = []
        
        if bboxes:
            # Compute comprehensive statistics from all bboxes for outlier detection
            valid_bboxes = [(x, y, w, h) for x, y, w, h in bboxes 
                           if isinstance((x, y, w, h), tuple) and h > 0 and w > 0]
            
            if valid_bboxes:
                areas = [w * h for x, y, w, h in valid_bboxes]
                aspect_ratios = [w / h for x, y, w, h in valid_bboxes]
                # Normalize aspect ratios to always be >= 1
                norm_aspects = [max(ar, 1.0/ar) if ar > 0 else 1.0 for ar in aspect_ratios]
                
                # Compute statistics for outlier detection
                bbox_stats = {
                    'median_area': float(np.median(areas)),
                    'mean_area': float(np.mean(areas)),
                    'area_std': float(np.std(areas)),
                    'q1_area': float(np.percentile(areas, 25)),
                    'q3_area': float(np.percentile(areas, 75)),
                    'median_aspect_ratio': float(np.median(norm_aspects)),
                    'mean_aspect_ratio': float(np.mean(norm_aspects)),
                    'aspect_ratio_std': float(np.std(norm_aspects)),
                    'q1_aspect': float(np.percentile(norm_aspects, 25)),
                    'q3_aspect': float(np.percentile(norm_aspects, 75)),
                    'num_pills': len(valid_bboxes)
                }
            else:
                bbox_stats = {
                    'median_area': 0, 'mean_area': 0, 'area_std': 0,
                    'q1_area': 0, 'q3_area': 0,
                    'median_aspect_ratio': 1.0, 'mean_aspect_ratio': 1.0,
                    'aspect_ratio_std': 0, 'q1_aspect': 1.0, 'q3_aspect': 1.0,
                    'num_pills': 0
                }
            
            # FIRST PASS: Collect blur and brightness metrics from all pills
            # Need this for same-image comparison (outlier detection)
            first_pass_metrics = []
            for i, bbox in enumerate(bboxes):
                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    continue
                x, y, w, h = bbox
                img_h, img_w = image.shape[:2]
                x, y = max(0, x), max(0, y)
                w, h = min(w, img_w - x), min(h, img_h - y)
                
                if w > 0 and h > 0:
                    crop = image[y:y+h, x:x+w]
                    if len(crop.shape) == 3:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = crop
                    
                    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    brightness = float(np.mean(gray))
                    
                    # Quick motion blur check
                    if gray.shape[0] > 10 and gray.shape[1] > 10:
                        motion_result = pill_analyzer.metrics_computer.compute_pill_motion_blur(gray)
                        motion_score = motion_result['motion_score']
                    else:
                        motion_score = 0.0
                    
                    first_pass_metrics.append({
                        'laplacian': lap_var,
                        'brightness': brightness,
                        'motion_score': motion_score
                    })
            
            # Compute same-image statistics for blur, brightness, motion
            if first_pass_metrics:
                lap_values = [m['laplacian'] for m in first_pass_metrics]
                bright_values = [m['brightness'] for m in first_pass_metrics]
                motion_values = [m['motion_score'] for m in first_pass_metrics]
                
                bbox_stats.update({
                    # Blur statistics (for detecting blurry pills vs sharp siblings)
                    'median_laplacian': float(np.median(lap_values)),
                    'mean_laplacian': float(np.mean(lap_values)),
                    'laplacian_std': float(np.std(lap_values)),
                    'q1_laplacian': float(np.percentile(lap_values, 25)),
                    'q3_laplacian': float(np.percentile(lap_values, 75)),
                    
                    # Brightness statistics (for detecting over/under exposed pills)
                    'median_brightness': float(np.median(bright_values)),
                    'mean_brightness': float(np.mean(bright_values)),
                    'brightness_std': float(np.std(bright_values)),
                    'q1_brightness': float(np.percentile(bright_values, 25)),
                    'q3_brightness': float(np.percentile(bright_values, 75)),
                    
                    # Motion blur statistics (for detecting pills still settling)
                    'median_motion': float(np.median(motion_values)),
                    'mean_motion': float(np.mean(motion_values)),
                    'motion_std': float(np.std(motion_values)),
                    'q1_motion': float(np.percentile(motion_values, 25)),
                    'q3_motion': float(np.percentile(motion_values, 75)),
                })
            
            # SECOND PASS: Full analysis with same-image statistics available
            for i, bbox in enumerate(bboxes):
                if not isinstance(bbox, tuple) or len(bbox) != 4:
                    continue
                pill_metrics = pill_analyzer.analyze_pill(image, bbox, i, bboxes, bbox_stats)
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
    on_side_count = 0
    
    for pm in pill_metrics_list:
        pill_issues = []
        
        # Stacking detection - use the computed boolean flag
        if pm.get('is_stacking', False):
            pill_issues.append("stacking")
            stacking_count += 1
        
        # Pill placed on side (edge-on instead of flat)
        if pm.get('is_on_side', False):
            pill_issues.append("pill_on_side")
            on_side_count += 1
        
        # Abnormal aspect ratio (might indicate partial pill or stacking)
        if (pm['aspect_ratio'] < 1/thresholds.pill_aspect_ratio_threshold or 
            pm['aspect_ratio'] > thresholds.pill_aspect_ratio_threshold):
            pill_issues.append("abnormal_aspect_ratio")
        
        # Size anomaly
        if pm['area_deviation'] > thresholds.pill_area_deviation_threshold:
            pill_issues.append("size_anomaly")
        
        # Pill blur - HYBRID: same-image outlier OR global threshold
        # This catches pills that are blurry compared to siblings (localized issue)
        # OR pills in images where everything is somewhat blurry but this one is worst
        is_blurry = pm.get('is_blur_outlier', False)
        if not is_blurry and pm['laplacian_variance'] < thresholds.pill_blur_threshold:
            is_blurry = True
        if is_blurry:
            pill_issues.append("pill_blur")
            blurry_pill_count += 1
        
        # Pill brightness - HYBRID: same-image outlier OR global threshold
        # Catches localized lighting issues where one pill is different from siblings
        is_bright_issue = pm.get('is_brightness_outlier', False)
        brightness_type = pm.get('brightness_outlier_type', '')
        
        if is_bright_issue and brightness_type == 'overexposed':
            pill_issues.append("pill_overexposed_vs_siblings")
            bright_pill_count += 1
        elif pm['mean_brightness'] > thresholds.pill_brightness_high:
            pill_issues.append("pill_overexposed")
            bright_pill_count += 1
        
        if is_bright_issue and brightness_type == 'underexposed':
            pill_issues.append("pill_underexposed_vs_siblings")
            dim_pill_count += 1
        elif pm['mean_brightness'] < thresholds.pill_brightness_low:
            pill_issues.append("pill_underexposed")
            dim_pill_count += 1
        
        # Motion blur on pill (pill was settling/moving when captured)
        # HYBRID: same-image outlier detection OR absolute detection
        has_motion = pm.get('has_motion_blur', False) or pm.get('is_motion_outlier', False)
        if has_motion:
            pill_issues.append("pill_motion_blur")
            motion_blur_pill_count += 1
        
        if pill_issues:
            bad_pill_count += 1
        
        pm['issues'] = pill_issues
    
    # Aggregate pill-level issues
    if stacking_count > 0:
        pill_level_issues.append(f"stacking_detected_{stacking_count}_pills")
    
    if on_side_count > 0:
        pill_level_issues.append(f"pills_on_side_{on_side_count}")
    
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
    bbox_extensions = {'.json', '.txt', '.csv', '.pkl', '.pickle'}
    
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
# DEBUG / DIAGNOSTIC MODE
# =============================================================================

def debug_analyze_image(image_path: str, bbox_path: str, output_dir: str):
    """
    Detailed debug analysis of a single image with visual output.
    """
    print(f"\n{'='*70}")
    print(f"DEBUG ANALYSIS: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        return
    
    img_height, img_width = image.shape[:2]
    print(f"Image size: {img_width} x {img_height}")
    
    # Load bboxes
    bboxes = load_bboxes(bbox_path)
    bboxes = convert_yolo_bboxes(bboxes, img_width, img_height)
    print(f"Loaded {len(bboxes)} bounding boxes")
    
    if not bboxes:
        print("WARNING: No bboxes loaded! Check bbox file format.")
        print(f"  Bbox path: {bbox_path}")
        print(f"  Exists: {os.path.exists(bbox_path)}")
        if os.path.exists(bbox_path):
            with open(bbox_path, 'rb') as f:
                raw_data = pickle.load(f)
            print(f"  Raw data type: {type(raw_data)}")
            print(f"  Raw data sample: {raw_data[:3] if isinstance(raw_data, list) else raw_data}")
        return
    
    # Show bbox samples
    print(f"\nBbox samples (first 3):")
    for i, bbox in enumerate(bboxes[:3]):
        print(f"  {i}: {bbox}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute image-level metrics
    print(f"\n--- IMAGE-LEVEL METRICS ---")
    metrics_computer = QualityMetricsComputer()
    
    # Individual metric computation with output
    lap_var = metrics_computer.compute_laplacian_variance(gray)
    print(f"Laplacian variance (blur):  {lap_var:.2f}")
    
    tenengrad = metrics_computer.compute_tenengrad(gray)
    print(f"Tenengrad (blur):           {tenengrad:.2f}")
    
    fft_ratio = metrics_computer.compute_fft_blur_metric(gray)
    print(f"FFT high-freq ratio:        {fft_ratio:.4f}")
    
    brightness = metrics_computer.compute_brightness_metrics(gray)
    print(f"Mean brightness:            {brightness['mean_brightness']:.2f}")
    print(f"Brightness std:             {brightness['brightness_std']:.2f}")
    print(f"Overexposed ratio:          {brightness['overexposed_ratio']*100:.2f}%")
    print(f"Underexposed ratio:         {brightness['underexposed_ratio']*100:.2f}%")
    
    uniformity = metrics_computer.compute_local_uniformity(gray)
    print(f"Brightness uniformity:      {uniformity['brightness_uniformity']:.4f}")
    print(f"Blur uniformity:            {uniformity['blur_uniformity']:.4f}")
    
    light = metrics_computer.compute_light_intrusion_metrics(image, bboxes)
    print(f"Light blob score:           {light['light_blob_score']*100:.4f}%")
    print(f"Light in pill region:       {light['light_in_pill_region']*100:.4f}%")
    print(f"Light in background:        {light['light_in_background']*100:.4f}%")
    
    # Compute pill-level metrics
    print(f"\n--- PILL-LEVEL ANALYSIS ---")
    
    valid_bboxes = [(x, y, w, h) for x, y, w, h in bboxes 
                   if isinstance((x, y, w, h), tuple) and h > 0 and w > 0]
    
    areas = [w * h for x, y, w, h in valid_bboxes]
    aspect_ratios = [w / h for x, y, w, h in valid_bboxes]
    norm_aspects = [max(ar, 1.0/ar) if ar > 0 else 1.0 for ar in aspect_ratios]
    
    print(f"\nBbox statistics:")
    print(f"  Area - min: {min(areas):.0f}, max: {max(areas):.0f}, median: {np.median(areas):.0f}, std: {np.std(areas):.0f}")
    print(f"  Aspect ratio - min: {min(norm_aspects):.2f}, max: {max(norm_aspects):.2f}, median: {np.median(norm_aspects):.2f}, std: {np.std(norm_aspects):.2f}")
    
    # Analyze a few pills
    pill_analyzer = PillAnalyzer()
    
    # Compute bbox stats
    bbox_stats = {
        'median_area': float(np.median(areas)),
        'mean_area': float(np.mean(areas)),
        'area_std': float(np.std(areas)),
        'q1_area': float(np.percentile(areas, 25)),
        'q3_area': float(np.percentile(areas, 75)),
        'median_aspect_ratio': float(np.median(norm_aspects)),
        'mean_aspect_ratio': float(np.mean(norm_aspects)),
        'aspect_ratio_std': float(np.std(norm_aspects)),
        'q1_aspect': float(np.percentile(norm_aspects, 25)),
        'q3_aspect': float(np.percentile(norm_aspects, 75)),
        'num_pills': len(valid_bboxes)
    }
    
    # First pass for blur/brightness stats
    first_pass_metrics = []
    for i, bbox in enumerate(bboxes[:min(len(bboxes), 200)]):  # Limit for speed
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        w, h = min(w, img_width - x), min(h, img_height - y)
        
        if w > 0 and h > 0:
            crop = image[y:y+h, x:x+w]
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
            
            lap_var = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
            bright = float(np.mean(crop_gray))
            
            first_pass_metrics.append({
                'laplacian': lap_var,
                'brightness': bright,
            })
    
    if first_pass_metrics:
        lap_values = [m['laplacian'] for m in first_pass_metrics]
        bright_values = [m['brightness'] for m in first_pass_metrics]
        
        bbox_stats.update({
            'median_laplacian': float(np.median(lap_values)),
            'laplacian_std': float(np.std(lap_values)),
            'q1_laplacian': float(np.percentile(lap_values, 25)),
            'q3_laplacian': float(np.percentile(lap_values, 75)),
            'median_brightness': float(np.median(bright_values)),
            'brightness_std': float(np.std(bright_values)),
            'q1_brightness': float(np.percentile(bright_values, 25)),
            'q3_brightness': float(np.percentile(bright_values, 75)),
        })
        
        print(f"\nPill metrics statistics:")
        print(f"  Laplacian - median: {bbox_stats['median_laplacian']:.2f}, std: {bbox_stats['laplacian_std']:.2f}")
        print(f"  Brightness - median: {bbox_stats['median_brightness']:.2f}, std: {bbox_stats['brightness_std']:.2f}")
    
    # Show detection thresholds being used
    print(f"\n--- DETECTION THRESHOLDS ---")
    print(f"Stacking: overlap > 20% of smaller pill")
    
    # IQR-based thresholds for on-side detection
    median_aspect = bbox_stats.get('median_aspect_ratio', 1.0)
    q1_aspect = bbox_stats.get('q1_aspect', 1.0)
    q3_aspect = bbox_stats.get('q3_aspect', 1.5)
    iqr_aspect = q3_aspect - q1_aspect
    aspect_upper_fence = q3_aspect + 1.5 * iqr_aspect
    
    median_area = bbox_stats.get('median_area', 0)
    q1_area = bbox_stats.get('q1_area', median_area * 0.7)
    q3_area = bbox_stats.get('q3_area', median_area * 1.3)
    iqr_area = q3_area - q1_area
    area_lower_fence = q1_area - 1.5 * iqr_area
    
    print(f"On-Side detection:")
    print(f"  Aspect ratio - median: {median_aspect:.2f}, Q1: {q1_aspect:.2f}, Q3: {q3_aspect:.2f}, IQR: {iqr_aspect:.2f}")
    print(f"  → Flagged if aspect > {aspect_upper_fence:.2f} (Q3 + 1.5*IQR)")
    print(f"  Area - median: {median_area:.0f}, Q1: {q1_area:.0f}, Q3: {q3_area:.0f}, IQR: {iqr_area:.0f}")
    print(f"  → Flagged if area < {area_lower_fence:.0f} (Q1 - 1.5*IQR)")
    
    # Analyze ALL pills and collect detailed data
    print(f"\n--- DETAILED PILL ANALYSIS ---")
    
    all_pill_data = []
    for i, bbox in enumerate(bboxes):
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        area = w * h
        aspect = max(w/h, h/w) if h > 0 and w > 0 else 1.0
        
        pm = pill_analyzer.analyze_pill(image, bbox, i, bboxes, bbox_stats)
        
        all_pill_data.append({
            'idx': i,
            'bbox': (x, y, w, h),
            'area': area,
            'aspect': aspect,
            'overlap': pm.overlap_ratio,
            'is_stacking': pm.is_stacking,
            'is_on_side': pm.is_on_side,
            'laplacian': pm.laplacian_variance,
            'brightness': pm.mean_brightness,
            'is_blur_outlier': pm.is_blur_outlier,
            'is_brightness_outlier': pm.is_brightness_outlier,
            'has_motion_blur': pm.has_motion_blur,
        })
    
    # Sort by aspect ratio to see distribution
    sorted_by_aspect = sorted(all_pill_data, key=lambda x: x['aspect'], reverse=True)
    
    print(f"\nTOP 10 HIGHEST ASPECT RATIO PILLS:")
    print(f"{'Idx':<6} {'Aspect':<8} {'Area':<8} {'Overlap':<8} {'Stacking':<10} {'OnSide':<8}")
    print("-" * 60)
    for p in sorted_by_aspect[:10]:
        print(f"{p['idx']:<6} {p['aspect']:<8.2f} {p['area']:<8} {p['overlap']:<8.2%} {str(p['is_stacking']):<10} {str(p['is_on_side']):<8}")
    
    # Sort by area (smallest first)
    sorted_by_area = sorted(all_pill_data, key=lambda x: x['area'])
    
    print(f"\nTOP 10 SMALLEST AREA PILLS:")
    print(f"{'Idx':<6} {'Area':<8} {'Aspect':<8} {'AreaRatio':<10} {'Stacking':<10} {'OnSide':<8}")
    print("-" * 65)
    for p in sorted_by_area[:10]:
        area_ratio = p['area'] / median_area if median_area > 0 else 0
        print(f"{p['idx']:<6} {p['area']:<8} {p['aspect']:<8.2f} {area_ratio:<10.2%} {str(p['is_stacking']):<10} {str(p['is_on_side']):<8}")
    
    # Sort by overlap (highest first) 
    sorted_by_overlap = sorted(all_pill_data, key=lambda x: x['overlap'], reverse=True)
    
    print(f"\nTOP 10 HIGHEST OVERLAP PILLS:")
    print(f"{'Idx':<6} {'Overlap':<10} {'Area':<8} {'Aspect':<8} {'Stacking':<10}")
    print("-" * 55)
    for p in sorted_by_overlap[:10]:
        print(f"{p['idx']:<6} {p['overlap']:<10.2%} {p['area']:<8} {p['aspect']:<8.2f} {str(p['is_stacking']):<10}")
    
    # Summary of flagged pills
    stacking_count = sum(1 for p in all_pill_data if p['is_stacking'])
    on_side_count = sum(1 for p in all_pill_data if p['is_on_side'])
    blur_count = sum(1 for p in all_pill_data if p['is_blur_outlier'])
    brightness_count = sum(1 for p in all_pill_data if p['is_brightness_outlier'])
    motion_count = sum(1 for p in all_pill_data if p['has_motion_blur'])
    
    print(f"\n--- FLAGGED PILLS SUMMARY ---")
    print(f"Total pills: {len(all_pill_data)}")
    print(f"  Stacking:    {stacking_count} ({100*stacking_count/len(all_pill_data):.1f}%)")
    print(f"  On-Side:     {on_side_count} ({100*on_side_count/len(all_pill_data):.1f}%)")
    print(f"  Blur:        {blur_count} ({100*blur_count/len(all_pill_data):.1f}%)")
    print(f"  Brightness:  {brightness_count} ({100*brightness_count/len(all_pill_data):.1f}%)")
    print(f"  Motion:      {motion_count} ({100*motion_count/len(all_pill_data):.1f}%)")
    
    # Show pills flagged as stacking with their overlap values
    if stacking_count > 0:
        print(f"\n  STACKING DETAILS:")
        stacking_pills = [p for p in all_pill_data if p['is_stacking']]
        for p in stacking_pills[:10]:
            print(f"    Pill {p['idx']}: overlap={p['overlap']:.2%}, bbox={p['bbox']}")
    
    # Show pills flagged as on-side with their values
    if on_side_count > 0:
        print(f"\n  ON-SIDE DETAILS:")
        on_side_pills = [p for p in all_pill_data if p['is_on_side']]
        for p in on_side_pills[:10]:
            area_ratio = p['area'] / median_area if median_area > 0 else 0
            print(f"    Pill {p['idx']}: aspect={p['aspect']:.2f} (thresh:{aspect_upper_fence:.2f}), area_ratio={area_ratio:.2%}, bbox={p['bbox']}")
    
    # Show detection thresholds
    print(f"\n--- DETECTION THRESHOLDS ---")
    iqr_area = bbox_stats['q3_area'] - bbox_stats['q1_area']
    iqr_aspect = bbox_stats['q3_aspect'] - bbox_stats['q1_aspect']
    print(f"  Stacking: bbox overlap > 20%")
    print(f"  On-side detection:")
    print(f"    Area lower fence: {bbox_stats['q1_area'] - 1.5 * iqr_area:.0f} (Q1 - 1.5*IQR)")
    print(f"    Aspect upper fence: {bbox_stats['q3_aspect'] + 1.5 * iqr_aspect:.2f} (Q3 + 1.5*IQR)")
    print(f"    Criteria: elongated outlier AND/OR small area outlier")
    
    # Analyze sample pills with detailed output
    print(f"\n--- SAMPLE PILL ANALYSIS (first 10) ---")
    
    issues_summary = {
        'stacking': 0,
        'on_side': 0,
        'blur_outlier': 0,
        'brightness_outlier': 0,
        'motion_blur': 0,
    }
    
    for i, bbox in enumerate(bboxes[:10]):
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        
        x, y, w, h = bbox
        area = w * h
        aspect = max(w/h, h/w) if h > 0 and w > 0 else 1.0
        
        pm = pill_analyzer.analyze_pill(image, bbox, i, bboxes, bbox_stats)
        
        # Check orientation manually for debug
        orientation = pill_analyzer.detect_pill_orientation(bbox, bbox_stats)
        
        issues = []
        details = []
        
        if pm.is_stacking:
            issues.append("STACKING")
            details.append(f"overlap={pm.overlap_ratio:.2f}")
            issues_summary['stacking'] += 1
        if pm.is_on_side:
            issues.append("ON_SIDE")
            details.append(f"area_ratio={orientation['area_ratio']:.2f}, aspect={aspect:.2f}")
            issues_summary['on_side'] += 1
        if pm.is_blur_outlier:
            issues.append("BLUR")
            issues_summary['blur_outlier'] += 1
        if pm.is_brightness_outlier:
            issues.append(f"BRIGHT({pm.brightness_outlier_type})")
            issues_summary['brightness_outlier'] += 1
        if pm.has_motion_blur:
            issues.append("MOTION")
            issues_summary['motion_blur'] += 1
        
        status = "⚠️ " + ", ".join(issues) if issues else "✓ OK"
        detail_str = f" [{', '.join(details)}]" if details else ""
        print(f"  Pill {i}: size={w}x{h}, area={area}, aspect={aspect:.2f} → {status}{detail_str}")
    
    # Full analysis
    print(f"\n--- FULL PILL ANALYSIS SUMMARY ---")
    all_issues = {k: 0 for k in issues_summary.keys()}
    
    for i, bbox in enumerate(bboxes):
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        pm = pill_analyzer.analyze_pill(image, bbox, i, bboxes, bbox_stats)
        
        if pm.is_stacking: all_issues['stacking'] += 1
        if pm.is_on_side: all_issues['on_side'] += 1
        if pm.is_blur_outlier: all_issues['blur_outlier'] += 1
        if pm.is_brightness_outlier: all_issues['brightness_outlier'] += 1
        if pm.has_motion_blur: all_issues['motion_blur'] += 1
    
    print(f"Total pills analyzed: {len(bboxes)}")
    for issue, count in all_issues.items():
        pct = 100 * count / len(bboxes) if bboxes else 0
        print(f"  {issue}: {count} ({pct:.1f}%)")
    
    # Create debug visualization
    debug_img = image.copy()
    
    # Add legend at top
    cv2.rectangle(debug_img, (10, 10), (350, 110), (255, 255, 255), -1)
    cv2.rectangle(debug_img, (10, 10), (350, 110), (0, 0, 0), 2)
    cv2.putText(debug_img, "Legend:", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.circle(debug_img, (30, 55), 8, (0, 255, 0), -1)
    cv2.putText(debug_img, "OK", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(debug_img, (100, 55), 8, (0, 0, 255), -1)
    cv2.putText(debug_img, "Stacking", (115, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(debug_img, (200, 55), 8, (255, 0, 255), -1)
    cv2.putText(debug_img, "On-Side", (215, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(debug_img, (30, 85), 8, (0, 165, 255), -1)
    cv2.putText(debug_img, "Blur", (45, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(debug_img, (100, 85), 8, (0, 255, 255), -1)
    cv2.putText(debug_img, "Brightness", (115, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.circle(debug_img, (200, 85), 8, (255, 165, 0), -1)
    cv2.putText(debug_img, "Motion", (215, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw bboxes with issue-specific colors
    stacking_pills = []
    on_side_pills = []
    
    for i, bbox in enumerate(bboxes):
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        pm = pill_analyzer.analyze_pill(image, bbox, i, bboxes, bbox_stats)
        
        # Determine color and label based on PRIORITY of issues
        color = (0, 255, 0)  # Default green
        label = ""
        thickness = 1
        
        if pm.is_stacking:
            color = (0, 0, 255)  # Red
            label = "STACK"
            thickness = 3
            stacking_pills.append((i, bbox, pm.overlap_ratio))
        elif pm.is_on_side:
            color = (255, 0, 255)  # Magenta
            label = "SIDE"
            thickness = 3
            on_side_pills.append((i, bbox, pm.aspect_ratio, pm.area_deviation))
        elif pm.is_blur_outlier or pm.has_motion_blur:
            color = (0, 165, 255)  # Orange
            label = "BLUR" if pm.is_blur_outlier else "MOTION"
            thickness = 2
        elif pm.is_brightness_outlier:
            color = (0, 255, 255)  # Yellow
            label = pm.brightness_outlier_type[:4].upper()
            thickness = 2
        
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, thickness)
        
        # Add small label
        if label:
            cv2.putText(debug_img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Print detailed info about flagged pills
    if stacking_pills:
        print(f"\n  STACKING PILLS ({len(stacking_pills)}):")
        for idx, bbox, overlap in stacking_pills[:5]:
            print(f"    Pill {idx}: bbox={bbox}, overlap={overlap:.2%}")
    
    if on_side_pills:
        print(f"\n  ON-SIDE PILLS ({len(on_side_pills)}):")
        print(f"    (Reference - median aspect: {bbox_stats.get('median_aspect_ratio', 0):.2f}, q3: {bbox_stats.get('q3_aspect', 0):.2f})")
        for idx, bbox, aspect, area_dev in on_side_pills[:5]:
            print(f"    Pill {idx}: bbox={bbox}, aspect={aspect:.2f}, area_dev={area_dev:.2%}")
    
    # Save debug image
    debug_path = os.path.join(output_dir, f"debug_{os.path.basename(image_path)}")
    cv2.imwrite(debug_path, debug_img)
    print(f"\nDebug visualization saved: {debug_path}")
    print("  Green = OK, Red = Stacking/OnSide, Orange = Blur/Motion, Yellow = Brightness")


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
  
  # Debug mode - analyze specific images with visual output
  python pill_quality_assessment.py --images ./images_consolidated --bboxes ./bboxes --output ./results --debug --debug-images img1.jpg img2.jpg
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
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with detailed output')
    parser.add_argument('--debug-images', nargs='*', default=None,
                       help='Specific image filenames to debug (used with --debug)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.isdir(args.images):
        print(f"Error: Images directory not found: {args.images}")
        return 1
    
    if not os.path.isdir(args.bboxes):
        print(f"Error: Bboxes directory not found: {args.bboxes}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Debug mode - analyze specific images with detailed output
    if args.debug:
        print("=" * 70)
        print("DEBUG MODE - Detailed Image Analysis")
        print("=" * 70)
        
        # Find image-bbox pairs
        pairs = find_image_bbox_pairs(args.images, args.bboxes)
        
        if args.debug_images:
            # Filter to specific images
            debug_set = set(args.debug_images)
            pairs = [(img, bbox) for img, bbox in pairs 
                    if os.path.basename(img) in debug_set or 
                       os.path.splitext(os.path.basename(img))[0] in debug_set]
        else:
            # Take first 5 images if none specified
            pairs = pairs[:5]
        
        if not pairs:
            print("No images found to debug!")
            return 1
        
        print(f"Debugging {len(pairs)} images...")
        
        for image_path, bbox_path in pairs:
            debug_analyze_image(image_path, bbox_path, args.output)
        
        print("\n" + "=" * 70)
        print("Debug complete! Check output directory for visualizations.")
        print("=" * 70)
        return 0
    
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