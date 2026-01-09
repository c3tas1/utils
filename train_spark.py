#!/usr/bin/env python3
"""
Prescription-Aware Pill Verification Training Script
=====================================================
Optimized for NVIDIA DGX Spark Personal AI Computer

DGX Spark Specs:
- GB10 Grace Blackwell Superchip
- 128GB unified LPDDR5x memory (shared CPU/GPU)
- 273 GB/s memory bandwidth
- 6,144 CUDA cores, 192 Tensor Cores
- 20 ARM cores (10 Cortex-X925 + 10 Cortex-A725)
- Up to 1 PFLOP FP4 AI performance

Optimizations for DGX Spark:
- Single GPU (no DDP overhead)
- Leverages large unified memory for bigger batches
- Memory-efficient data loading with prefetching
- FP16/BF16 mixed precision for Tensor Cores
- Gradient checkpointing for memory efficiency
- Optimized for lower memory bandwidth (273 GB/s)

Usage:
    # Build index first (run once)
    python train_dgx_spark.py --build-index --data-dir /path/to/data
    
    # Train
    python train_dgx_spark.py \
        --data-dir /path/to/data \
        --output-dir /path/to/output \
        --epochs 100 \
        --batch-size 16

Author: Claude (Anthropic)
"""

import os
import sys
import re
import csv
import json
import math
import random
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import gc

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torchvision import transforms, models
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Configuration - Optimized for DGX Spark
# =============================================================================

@dataclass
class DGXSparkConfig:
    """Training configuration optimized for DGX Spark's unified memory architecture."""
    
    # Data paths
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Model architecture
    num_classes: int = 1000
    pill_embed_dim: int = 512
    max_pills_per_prescription: int = 200
    min_pills_per_prescription: int = 5
    
    # Training hyperparameters - optimized for DGX Spark
    # With 128GB unified memory, we can use larger batches
    epochs: int = 100
    batch_size: int = 16  # Larger batch size thanks to 128GB memory
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Loss weights
    script_loss_weight: float = 1.0
    pill_loss_weight: float = 0.5
    
    # Error injection
    error_rate: float = 0.5
    
    # DGX Spark optimizations
    use_amp: bool = True  # Use mixed precision for Tensor Cores
    amp_dtype: str = "bfloat16"  # BF16 works well on Blackwell
    use_gradient_checkpointing: bool = True  # Save memory
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Data loading - optimized for unified memory
    num_workers: int = 8  # Use ARM efficiency cores for data loading
    pin_memory: bool = False  # Not needed with unified memory
    prefetch_factor: int = 4  # Aggressive prefetching
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_n_steps: int = 10
    collage_every_n_epochs: int = 1
    num_collage_samples: int = 16
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_from: Optional[str] = None


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: str) -> logging.Logger:
    """Setup logging to console and file."""
    logger = logging.getLogger("PillVerifier-DGXSpark")
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# =============================================================================
# DGX Spark Device Setup
# =============================================================================

def setup_dgx_spark_device(logger: logging.Logger) -> torch.device:
    """Setup and optimize for DGX Spark."""
    
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device('cpu')
    
    device = torch.device('cuda:0')
    
    # Get device properties
    props = torch.cuda.get_device_properties(0)
    logger.info(f"Device: {props.name}")
    logger.info(f"Total Memory: {props.total_memory / 1024**3:.1f} GB")
    logger.info(f"CUDA Capability: {props.major}.{props.minor}")
    
    # Check if this looks like a DGX Spark (Blackwell GPU)
    if "Blackwell" in props.name or "GB10" in props.name:
        logger.info("✓ DGX Spark (Blackwell) detected")
    else:
        logger.info(f"Running on: {props.name}")
    
    # Enable optimizations
    # TF32 for faster matmul (if available)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # cuDNN benchmarking
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Memory management for unified memory
    # With unified memory, we don't need aggressive caching
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    logger.info("✓ Device optimizations enabled")
    
    return device


def get_amp_dtype(config: DGXSparkConfig) -> torch.dtype:
    """Get the appropriate AMP dtype for DGX Spark."""
    if config.amp_dtype == "bfloat16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float16


# =============================================================================
# Fast Dataset Indexing
# =============================================================================

class DatasetIndexer:
    """Fast CSV-based indexing for large-scale pill image datasets."""
    
    # Updated pattern to match: {ndc}_{prescription_id}_patch_{patch_no}.jpg
    FILENAME_PATTERN = re.compile(r'^(.+)_(.+)_patch_(\d+)\.jpg$', re.IGNORECASE)
    
    @classmethod
    def build_index(cls, data_dir: str, output_file: str, splits: List[str] = ['train', 'valid']) -> Dict[str, int]:
        """Build CSV index for all images."""
        print(f"Building index for {data_dir}...")
        
        stats = defaultdict(int)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['split', 'ndc', 'prescription_id', 'patch_no', 'relative_path'])
            
            for split in splits:
                split_dir = Path(data_dir) / split
                if not split_dir.exists():
                    print(f"  Warning: {split_dir} does not exist, skipping")
                    continue
                
                ndc_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
                print(f"  Processing {split}: {len(ndc_dirs)} NDC directories")
                
                for ndc_dir in tqdm(ndc_dirs, desc=f"Indexing {split}"):
                    ndc = ndc_dir.name
                    
                    for img_path in ndc_dir.glob('*.jpg'):
                        match = cls.FILENAME_PATTERN.match(img_path.name)
                        if match:
                            file_ndc, prescription_id, patch_no = match.groups()
                            relative_path = str(img_path.relative_to(data_dir))
                            
                            writer.writerow([split, ndc, prescription_id, int(patch_no), relative_path])
                            stats[f'{split}_images'] += 1
                
                stats[f'{split}_ndcs'] = len(ndc_dirs)
        
        # Count unique prescriptions
        prescriptions_per_split = defaultdict(set)
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prescriptions_per_split[row['split']].add(row['prescription_id'])
        
        for split in splits:
            stats[f'{split}_prescriptions'] = len(prescriptions_per_split[split])
        
        print(f"\nIndex built: {output_file}")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value:,}")
        
        return dict(stats)
    
    @classmethod
    def load_index(cls, index_file: str) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
        """Load pre-built CSV index."""
        data_by_split = defaultdict(list)
        all_ndcs = set()
        
        with open(index_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_by_split[row['split']].append({
                    'ndc': row['ndc'],
                    'prescription_id': row['prescription_id'],
                    'patch_no': int(row['patch_no']),
                    'relative_path': row['relative_path']
                })
                all_ndcs.add(row['ndc'])
        
        class_to_idx = {ndc: idx for idx, ndc in enumerate(sorted(all_ndcs))}
        
        return dict(data_by_split), class_to_idx


# =============================================================================
# Dataset Classes
# =============================================================================

class PrescriptionDataset(Dataset):
    """Prescription-level dataset optimized for DGX Spark's unified memory."""
    
    def __init__(
        self,
        data_dir: str,
        split_data: List[dict],
        class_to_idx: Dict[str, int],
        transform=None,
        max_pills: int = 200,
        min_pills: int = 5,
        error_rate: float = 0.0,
        is_training: bool = True,
        cache_images: bool = False  # Can enable with 128GB memory
    ):
        self.data_dir = Path(data_dir)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.num_classes = len(class_to_idx)
        self.transform = transform or self._default_transform(is_training)
        self.max_pills = max_pills
        self.min_pills = min_pills
        self.error_rate = error_rate
        self.is_training = is_training
        self.cache_images = cache_images
        
        # Image cache (leverage unified memory)
        self.image_cache = {} if cache_images else None
        
        # Group by prescription
        self.prescriptions = self._group_by_prescription(split_data)
        self.prescription_ids = list(self.prescriptions.keys())
        
        # Build pill library for error injection
        if error_rate > 0:
            self.pill_library = self._build_pill_library()
        else:
            self.pill_library = {}
    
    def _group_by_prescription(self, data: List[dict]) -> Dict[str, List[dict]]:
        """Group pills by prescription ID."""
        prescriptions = defaultdict(list)
        
        for item in data:
            prescriptions[item['prescription_id']].append(item)
        
        filtered = {}
        for rx_id, pills in prescriptions.items():
            if self.min_pills <= len(pills) <= self.max_pills:
                filtered[rx_id] = sorted(pills, key=lambda x: x['patch_no'])
        
        return filtered
    
    def _build_pill_library(self) -> Dict[str, List[str]]:
        """Build library of pill paths per NDC."""
        library = defaultdict(list)
        for rx_id, pills in self.prescriptions.items():
            for pill in pills:
                library[pill['ndc']].append(pill['relative_path'])
        return dict(library)
    
    def _default_transform(self, is_training: bool):
        if is_training:
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_image(self, relative_path: str) -> Image.Image:
        """Load image with optional caching."""
        if self.cache_images and relative_path in self.image_cache:
            return self.image_cache[relative_path].copy()
        
        img_path = self.data_dir / relative_path
        img = Image.open(img_path).convert('RGB')
        
        if self.cache_images:
            self.image_cache[relative_path] = img.copy()
        
        return img
    
    def _get_random_wrong_pill(self, correct_ndc: str) -> Tuple[str, str, int]:
        """Get a random pill from a different NDC."""
        available_ndcs = [ndc for ndc in self.pill_library.keys() if ndc != correct_ndc]
        if not available_ndcs:
            return random.choice(self.pill_library[correct_ndc]), correct_ndc, self.class_to_idx[correct_ndc]
        
        wrong_ndc = random.choice(available_ndcs)
        wrong_path = random.choice(self.pill_library[wrong_ndc])
        wrong_class_idx = self.class_to_idx[wrong_ndc]
        return wrong_path, wrong_ndc, wrong_class_idx
    
    def _inject_errors(self, pills: List[dict]) -> Tuple[List[dict], List[int], str]:
        """Inject errors into a prescription."""
        error_type = random.choices(
            ['single', 'few', 'many', 'all'],
            weights=[0.5, 0.3, 0.15, 0.05]
        )[0]
        
        n_pills = len(pills)
        modified_pills = [p.copy() for p in pills]
        
        if error_type == 'single':
            n_wrong = 1
        elif error_type == 'few':
            n_wrong = random.randint(2, min(5, max(2, n_pills // 4)))
        elif error_type == 'many':
            n_wrong = random.randint(max(2, n_pills // 4), max(3, n_pills // 2))
        else:
            n_wrong = n_pills
        
        wrong_indices = random.sample(range(n_pills), min(n_wrong, n_pills))
        
        for idx in wrong_indices:
            original_ndc = modified_pills[idx]['ndc']
            wrong_path, wrong_ndc, wrong_class_idx = self._get_random_wrong_pill(original_ndc)
            
            modified_pills[idx] = {
                'relative_path': wrong_path,
                'ndc': wrong_ndc,
                'patch_no': modified_pills[idx]['patch_no'],
                'prescription_id': modified_pills[idx]['prescription_id'],
                'is_wrong': True,
                'original_ndc': original_ndc
            }
        
        return modified_pills, wrong_indices, error_type
    
    def get_prescription_composition(self, prescription_id: str) -> Dict[str, int]:
        """Get expected NDC composition."""
        pills = self.prescriptions[prescription_id]
        composition = defaultdict(int)
        for pill in pills:
            composition[pill['ndc']] += 1
        return dict(composition)
    
    def __len__(self):
        return len(self.prescription_ids)
    
    def __getitem__(self, idx):
        prescription_id = self.prescription_ids[idx]
        original_pills = self.prescriptions[prescription_id]
        
        inject_error = self.is_training and random.random() < self.error_rate
        
        if inject_error:
            pills, wrong_indices, error_type = self._inject_errors(original_pills)
            is_correct = False
        else:
            pills = original_pills
            wrong_indices = []
            error_type = 'none'
            is_correct = True
        
        images = []
        actual_labels = []
        paths = []
        
        for pill in pills:
            try:
                img = self._load_image(pill['relative_path'])
                if self.transform:
                    img = self.transform(img)
                images.append(img)
                actual_labels.append(self.class_to_idx[pill['ndc']])
                paths.append(pill['relative_path'])
            except Exception as e:
                img = torch.zeros(3, 224, 224)
                images.append(img)
                actual_labels.append(self.class_to_idx.get(pill['ndc'], 0))
                paths.append(pill['relative_path'])
        
        images = torch.stack(images)
        actual_labels = torch.tensor(actual_labels, dtype=torch.long)
        
        composition = self.get_prescription_composition(prescription_id)
        expected_ndcs = [self.class_to_idx[ndc] for ndc in composition.keys()]
        expected_counts = list(composition.values())
        
        pill_correct = torch.ones(len(pills), dtype=torch.float)
        for idx in wrong_indices:
            pill_correct[idx] = 0.0
        
        return {
            'images': images,
            'actual_labels': actual_labels,
            'prescription_id': prescription_id,
            'num_pills': len(pills),
            'expected_ndcs': torch.tensor(expected_ndcs, dtype=torch.long),
            'expected_counts': torch.tensor(expected_counts, dtype=torch.float),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': pill_correct,
            'wrong_indices': wrong_indices,
            'error_type': error_type,
            'paths': paths
        }


def prescription_collate_fn(batch: List[dict]) -> dict:
    """Custom collate for variable-length prescriptions."""
    max_pills = max(item['num_pills'] for item in batch)
    max_expected = max(len(item['expected_ndcs']) for item in batch)
    batch_size = len(batch)
    
    C, H, W = batch[0]['images'].shape[1:]
    
    images_padded = torch.zeros(batch_size, max_pills, C, H, W)
    actual_labels_padded = torch.full((batch_size, max_pills), -1, dtype=torch.long)
    pill_correct_padded = torch.ones(batch_size, max_pills)
    pill_mask = torch.ones(batch_size, max_pills, dtype=torch.bool)
    
    expected_ndcs_padded = torch.zeros(batch_size, max_expected, dtype=torch.long)
    expected_counts_padded = torch.zeros(batch_size, max_expected, dtype=torch.float)
    expected_mask = torch.ones(batch_size, max_expected, dtype=torch.bool)
    
    is_correct = torch.zeros(batch_size)
    num_pills = torch.zeros(batch_size, dtype=torch.long)
    prescription_ids = []
    all_paths = []
    all_wrong_indices = []
    
    for i, item in enumerate(batch):
        n = item['num_pills']
        n_expected = len(item['expected_ndcs'])
        
        images_padded[i, :n] = item['images']
        actual_labels_padded[i, :n] = item['actual_labels']
        pill_correct_padded[i, :n] = item['pill_correct']
        pill_mask[i, :n] = False
        
        expected_ndcs_padded[i, :n_expected] = item['expected_ndcs']
        expected_counts_padded[i, :n_expected] = item['expected_counts']
        expected_mask[i, :n_expected] = False
        
        is_correct[i] = item['is_correct']
        num_pills[i] = n
        prescription_ids.append(item['prescription_id'])
        all_paths.append(item['paths'])
        all_wrong_indices.append(item['wrong_indices'])
    
    return {
        'images': images_padded,
        'actual_labels': actual_labels_padded,
        'pill_correct': pill_correct_padded,
        'pill_mask': pill_mask,
        'expected_ndcs': expected_ndcs_padded,
        'expected_counts': expected_counts_padded,
        'expected_mask': expected_mask,
        'is_correct': is_correct,
        'num_pills': num_pills,
        'prescription_ids': prescription_ids,
        'paths': all_paths,
        'wrong_indices': all_wrong_indices
    }


# =============================================================================
# Model Architecture
# =============================================================================

class PrescriptionAwareVerifier(nn.Module):
    """Prescription-aware verification model using Set Transformer."""
    
    def __init__(
        self,
        pill_embed_dim: int = 512,
        ndc_vocab_size: int = 1000,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        self.ndc_embedding = nn.Embedding(ndc_vocab_size, hidden_dim)
        
        self.prescription_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )
        
        self.pill_projector = nn.Sequential(
            nn.Linear(pill_embed_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        self.cross_attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(2)
        ])
        
        self.pill_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers
        )
        
        self.summary_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        
        self.script_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        self.pill_anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def _cross_attention_block(self, x, context, context_mask, layer_idx):
        attn_out, _ = self.cross_attention_layers[layer_idx](
            query=x,
            key=context,
            value=context,
            key_padding_mask=context_mask
        )
        return self.cross_attention_norms[layer_idx](x + attn_out)
    
    def forward(
        self,
        pill_embeddings: torch.Tensor,
        pill_confidences: torch.Tensor,
        prescription_ndcs: torch.Tensor,
        prescription_counts: torch.Tensor,
        pill_mask: Optional[torch.Tensor] = None,
        prescription_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = pill_embeddings.size(0)
        
        rx_embed = self.ndc_embedding(prescription_ndcs)
        count_scale = prescription_counts / (prescription_counts.sum(dim=1, keepdim=True) + 1e-8)
        rx_embed = rx_embed + count_scale * 0.1
        
        if self.use_gradient_checkpointing and self.training:
            rx_encoded = checkpoint(
                self.prescription_encoder,
                rx_embed,
                use_reentrant=False
            )
        else:
            rx_encoded = self.prescription_encoder(
                rx_embed,
                src_key_padding_mask=prescription_mask
            )
        
        pill_features = torch.cat([pill_embeddings, pill_confidences], dim=-1)
        pill_projected = self.pill_projector(pill_features)
        
        for i in range(len(self.cross_attention_layers)):
            pill_projected = self._cross_attention_block(
                pill_projected, rx_encoded, prescription_mask, i
            )
        
        if self.use_gradient_checkpointing and self.training:
            pill_encoded = checkpoint(
                self.pill_encoder,
                pill_projected,
                use_reentrant=False
            )
        else:
            pill_encoded = self.pill_encoder(
                pill_projected,
                src_key_padding_mask=pill_mask
            )
        
        pill_anomaly = self.pill_anomaly_head(pill_encoded).squeeze(-1)
        
        summary = self.summary_token.expand(B, -1, -1)
        set_summary, _ = self.pool_attention(
            query=summary,
            key=pill_encoded,
            value=pill_encoded,
            key_padding_mask=pill_mask
        )
        set_summary = self.pool_norm(summary + set_summary)
        
        script_logits = self.script_classifier(set_summary.squeeze(1))
        
        return script_logits, pill_anomaly


class EmbeddingExtractor(nn.Module):
    """Wrapper for pretrained ResNet to extract pill embeddings."""
    
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        num_classes: int = 1000,
        freeze: bool = True
    ):
        super().__init__()
        
        if pretrained_path and os.path.exists(pretrained_path):
            self.resnet = models.resnet34(weights=None)
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.resnet.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        else:
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            print("Using ImageNet pretrained weights")
        
        self.embed_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False
            self.resnet.eval()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = x.shape
        
        if len(original_shape) == 5:
            B, N, C, H, W = original_shape
            x = x.view(B * N, C, H, W)
        
        embeddings = self.resnet(x)
        logits = self.classifier(embeddings)
        
        if len(original_shape) == 5:
            embeddings = embeddings.view(B, N, -1)
            logits = logits.view(B, N, -1)
        
        return embeddings, logits


# =============================================================================
# Training Utilities
# =============================================================================

class AverageMeter:
    """Computes and stores running averages."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        if n <= 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class LRScheduler:
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        base_lr: Optional[float] = None
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self.current_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def compute_metrics(
    script_logits: torch.Tensor,
    pill_anomaly: torch.Tensor,
    script_labels: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor
) -> Dict[str, float]:
    """Compute training/validation metrics."""
    
    with torch.no_grad():
        script_probs = torch.sigmoid(script_logits.squeeze(-1))
        script_preds = (script_probs > 0.5).float()
        script_acc = (script_preds == script_labels).float().mean().item()
        
        pill_probs = torch.sigmoid(pill_anomaly)
        pill_preds = (pill_probs > 0.5).float()
        pill_labels_binary = 1 - pill_labels
        
        valid_mask = ~pill_mask
        if valid_mask.any():
            pill_acc = (
                (pill_preds == pill_labels_binary)[valid_mask].float().mean().item()
            )
        else:
            pill_acc = 0.0
    
    return {
        'script_acc': script_acc,
        'pill_acc': pill_acc
    }


def compute_loss(
    script_logits: torch.Tensor,
    pill_anomaly: torch.Tensor,
    script_labels: torch.Tensor,
    pill_labels: torch.Tensor,
    pill_mask: torch.Tensor,
    script_weight: float = 1.0,
    pill_weight: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute combined loss."""
    
    script_loss = F.binary_cross_entropy_with_logits(
        script_logits.squeeze(-1), script_labels
    )
    
    pill_labels_binary = 1 - pill_labels
    pill_loss_raw = F.binary_cross_entropy_with_logits(
        pill_anomaly, pill_labels_binary, reduction='none'
    )
    
    valid_mask = ~pill_mask
    if valid_mask.any():
        pill_loss = (pill_loss_raw * valid_mask.float()).sum() / valid_mask.sum()
    else:
        pill_loss = torch.tensor(0.0, device=script_loss.device)
    
    total_loss = script_weight * script_loss + pill_weight * pill_loss
    
    return total_loss, {
        'total': total_loss.item(),
        'script': script_loss.item(),
        'pill': pill_loss.item()
    }


# =============================================================================
# Visualization
# =============================================================================

def create_validation_collage(
    model: nn.Module,
    embedding_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_path: str,
    num_samples: int = 16,
    epoch: int = 0,
    amp_dtype: torch.dtype = torch.float16
):
    """Create visualization collage."""
    
    model.eval()
    embedding_model.eval()
    
    samples = []
    
    with torch.no_grad():
        for batch in val_loader:
            if len(samples) >= num_samples:
                break
            
            images = batch['images'].to(device)
            pill_mask = batch['pill_mask'].to(device)
            expected_ndcs = batch['expected_ndcs'].to(device)
            expected_counts = batch['expected_counts'].to(device)
            expected_mask = batch['expected_mask'].to(device)
            is_correct = batch['is_correct'].to(device)
            pill_correct = batch['pill_correct'].to(device)
            num_pills = batch['num_pills']
            prescription_ids = batch['prescription_ids']
            wrong_indices = batch['wrong_indices']
            
            B, max_pills, C, H, W = images.shape
            
            with autocast(dtype=amp_dtype):
                images_flat = images.view(B * max_pills, C, H, W)
                embeddings_flat, logits_flat = embedding_model(images_flat)
                pill_embeddings = embeddings_flat.view(B, max_pills, -1)
                pill_logits = logits_flat.view(B, max_pills, -1)
                
                pill_probs = F.softmax(pill_logits.float(), dim=-1)
                pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
                
                script_logits, pill_anomaly = model(
                    pill_embeddings=pill_embeddings.float(),
                    pill_confidences=pill_confidences.float(),
                    prescription_ndcs=expected_ndcs,
                    prescription_counts=expected_counts.unsqueeze(-1),
                    pill_mask=pill_mask,
                    prescription_mask=expected_mask
                )
            
            script_probs = torch.sigmoid(script_logits.squeeze(-1).float())
            pill_anomaly_probs = torch.sigmoid(pill_anomaly.float())
            
            for i in range(B):
                if len(samples) >= num_samples:
                    break
                
                n = num_pills[i].item()
                
                sample_images = images[i, :n].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                sample_images = sample_images * std + mean
                sample_images = sample_images.clamp(0, 1)
                
                samples.append({
                    'images': sample_images,
                    'num_pills': n,
                    'prescription_id': prescription_ids[i],
                    'is_correct_gt': is_correct[i].item(),
                    'script_prob': script_probs[i].item(),
                    'pill_anomaly_probs': pill_anomaly_probs[i, :n].cpu().numpy(),
                    'pill_correct_gt': pill_correct[i, :n].cpu().numpy(),
                    'wrong_indices': wrong_indices[i]
                })
    
    # Create collage
    n_cols = 4
    n_rows = (len(samples) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample in enumerate(samples):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        n_pills = min(sample['num_pills'], 16)
        grid_size = int(np.ceil(np.sqrt(n_pills)))
        
        pill_grid = np.ones((grid_size * 64, grid_size * 64, 3))
        
        for p_idx in range(n_pills):
            r, c = p_idx // grid_size, p_idx % grid_size
            pill_img = sample['images'][p_idx].permute(1, 2, 0).numpy()
            pill_img = np.array(Image.fromarray(
                (pill_img * 255).astype(np.uint8)
            ).resize((60, 60)))
            
            anomaly_prob = sample['pill_anomaly_probs'][p_idx]
            is_wrong_gt = p_idx in sample['wrong_indices']
            
            pill_padded = np.ones((64, 64, 3)) * (1 - anomaly_prob)
            pill_padded[2:62, 2:62] = pill_img / 255.0
            
            if is_wrong_gt:
                border_color = [1, 0, 0]
                pill_padded[:2, :] = border_color
                pill_padded[-2:, :] = border_color
                pill_padded[:, :2] = border_color
                pill_padded[:, -2:] = border_color
            
            pill_grid[r*64:(r+1)*64, c*64:(c+1)*64] = pill_padded
        
        ax.imshow(pill_grid)
        
        gt_label = "CORRECT" if sample['is_correct_gt'] else "WRONG"
        pred_label = "CORRECT" if sample['script_prob'] > 0.5 else "WRONG"
        color = 'green' if (sample['script_prob'] > 0.5) == sample['is_correct_gt'] else 'red'
        
        ax.set_title(
            f"Rx: {sample['prescription_id'][:10]}...\n"
            f"GT: {gt_label} | Pred: {pred_label} ({sample['script_prob']:.2f})\n"
            f"Pills: {sample['num_pills']} | Wrong: {len(sample['wrong_indices'])}",
            fontsize=10,
            color=color
        )
        ax.axis('off')
    
    for idx in range(len(samples), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Validation Results - Epoch {epoch} (DGX Spark)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    embedding_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    best_metric: float,
    config: DGXSparkConfig,
    output_dir: str,
    is_best: bool = False
):
    """Save training checkpoint."""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'embedding_model_state_dict': embedding_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
        'config': config.__dict__
    }
    
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint_data, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint_data, best_path)
    
    # Clean up old checkpoints
    checkpoints = sorted(
        [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    
    while len(checkpoints) > config.keep_last_n_checkpoints:
        old_checkpoint = checkpoints.pop(0)
        os.remove(os.path.join(output_dir, old_checkpoint))


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    embedding_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device
) -> Tuple[int, float]:
    """Load training checkpoint."""
    checkpoint_data = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint_data['model_state_dict'])
    embedding_model.load_state_dict(checkpoint_data['embedding_model_state_dict'])
    optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint_data['scaler_state_dict'])
    
    return checkpoint_data['epoch'], checkpoint_data['best_metric']


# =============================================================================
# Main Training Loop - Optimized for DGX Spark
# =============================================================================

def train_epoch(
    model: nn.Module,
    embedding_model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: DGXSparkConfig,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    amp_dtype: torch.dtype
) -> Dict[str, float]:
    """Train for one epoch on DGX Spark."""
    
    model.train()
    embedding_model.eval()
    
    loss_meter = AverageMeter('Loss')
    script_loss_meter = AverageMeter('Script Loss')
    pill_loss_meter = AverageMeter('Pill Loss')
    script_acc_meter = AverageMeter('Script Acc')
    pill_acc_meter = AverageMeter('Pill Acc')
    
    pbar = tqdm(
        total=len(train_loader),
        desc=f'Epoch {epoch}',
        dynamic_ncols=True
    )
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(train_loader):
        # Move to device (unified memory makes this fast)
        images = batch['images'].to(device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(device, non_blocking=True)
        expected_ndcs = batch['expected_ndcs'].to(device, non_blocking=True)
        expected_counts = batch['expected_counts'].to(device, non_blocking=True)
        expected_mask = batch['expected_mask'].to(device, non_blocking=True)
        is_correct = batch['is_correct'].to(device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(device, non_blocking=True)
        
        B, max_pills, C, H, W = images.shape
        
        sync_gradients = (step + 1) % config.gradient_accumulation_steps == 0
        
        with autocast(dtype=amp_dtype, enabled=config.use_amp):
            # Extract embeddings (frozen)
            with torch.no_grad():
                images_flat = images.view(B * max_pills, C, H, W)
                embeddings_flat, logits_flat = embedding_model(images_flat)
                pill_embeddings = embeddings_flat.view(B, max_pills, -1)
                pill_logits = logits_flat.view(B, max_pills, -1)
                
                pill_probs = F.softmax(pill_logits.float(), dim=-1)
                pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
            
            # Forward through verifier
            script_logits, pill_anomaly = model(
                pill_embeddings=pill_embeddings,
                pill_confidences=pill_confidences,
                prescription_ndcs=expected_ndcs,
                prescription_counts=expected_counts.unsqueeze(-1),
                pill_mask=pill_mask,
                prescription_mask=expected_mask
            )
            
            loss, loss_components = compute_loss(
                script_logits, pill_anomaly,
                is_correct, pill_correct, pill_mask,
                config.script_loss_weight, config.pill_loss_weight
            )
            
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if sync_gradients:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        metrics = compute_metrics(
            script_logits, pill_anomaly,
            is_correct, pill_correct, pill_mask
        )
        
        loss_meter.update(loss_components['total'], B)
        script_loss_meter.update(loss_components['script'], B)
        pill_loss_meter.update(loss_components['pill'], B)
        script_acc_meter.update(metrics['script_acc'], B)
        pill_acc_meter.update(metrics['pill_acc'], B)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'script_loss': f'{script_loss_meter.avg:.4f}',
            'pill_loss': f'{pill_loss_meter.avg:.4f}',
            'script_acc': f'{script_acc_meter.avg:.4f}',
            'pill_acc': f'{pill_acc_meter.avg:.4f}'
        })
        pbar.update(1)
    
    pbar.close()
    
    return {
        'loss': loss_meter.avg,
        'script_acc': script_acc_meter.avg,
        'pill_acc': pill_acc_meter.avg
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    embedding_model: nn.Module,
    val_loader: DataLoader,
    config: DGXSparkConfig,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    amp_dtype: torch.dtype
) -> Dict[str, float]:
    """Validation loop."""
    
    model.eval()
    embedding_model.eval()
    
    loss_meter = AverageMeter('Loss')
    script_acc_meter = AverageMeter('Script Acc')
    pill_acc_meter = AverageMeter('Pill Acc')
    
    pbar = tqdm(
        total=len(val_loader),
        desc=f'Val {epoch}',
        dynamic_ncols=True
    )
    
    for batch in val_loader:
        images = batch['images'].to(device, non_blocking=True)
        pill_mask = batch['pill_mask'].to(device, non_blocking=True)
        expected_ndcs = batch['expected_ndcs'].to(device, non_blocking=True)
        expected_counts = batch['expected_counts'].to(device, non_blocking=True)
        expected_mask = batch['expected_mask'].to(device, non_blocking=True)
        is_correct = batch['is_correct'].to(device, non_blocking=True)
        pill_correct = batch['pill_correct'].to(device, non_blocking=True)
        
        B, max_pills, C, H, W = images.shape
        
        with autocast(dtype=amp_dtype, enabled=config.use_amp):
            images_flat = images.view(B * max_pills, C, H, W)
            embeddings_flat, logits_flat = embedding_model(images_flat)
            pill_embeddings = embeddings_flat.view(B, max_pills, -1)
            pill_logits = logits_flat.view(B, max_pills, -1)
            
            pill_probs = F.softmax(pill_logits.float(), dim=-1)
            pill_confidences = pill_probs.max(dim=-1).values.unsqueeze(-1)
            
            script_logits, pill_anomaly = model(
                pill_embeddings=pill_embeddings,
                pill_confidences=pill_confidences,
                prescription_ndcs=expected_ndcs,
                prescription_counts=expected_counts.unsqueeze(-1),
                pill_mask=pill_mask,
                prescription_mask=expected_mask
            )
            
            loss, _ = compute_loss(
                script_logits, pill_anomaly,
                is_correct, pill_correct, pill_mask,
                config.script_loss_weight, config.pill_loss_weight
            )
        
        metrics = compute_metrics(
            script_logits, pill_anomaly,
            is_correct, pill_correct, pill_mask
        )
        
        loss_meter.update(loss.item(), B)
        script_acc_meter.update(metrics['script_acc'], B)
        pill_acc_meter.update(metrics['pill_acc'], B)
        
        pbar.set_postfix({
            'val_loss': f'{loss_meter.avg:.4f}',
            'script_acc': f'{script_acc_meter.avg:.4f}',
            'pill_acc': f'{pill_acc_meter.avg:.4f}'
        })
        pbar.update(1)
    
    pbar.close()
    
    return {
        'loss': loss_meter.avg,
        'script_acc': script_acc_meter.avg,
        'pill_acc': pill_acc_meter.avg
    }


def main():
    """Main training function for DGX Spark."""
    
    parser = argparse.ArgumentParser(description='Train Pill Verifier on DGX Spark')
    
    parser.add_argument('--data-dir', type=str, required=True, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--index-file', type=str, default='dataset_index.csv', help='Index file')
    parser.add_argument('--build-index', action='store_true', help='Build index and exit')
    
    parser.add_argument('--num-classes', type=int, default=1000, help='Number of NDCs')
    parser.add_argument('--resnet-checkpoint', type=str, default=None, help='Pretrained ResNet')
    
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--error-rate', type=float, default=0.5, help='Error rate')
    parser.add_argument('--min-pills', type=int, default=5, help='Min pills per prescription')
    parser.add_argument('--max-pills', type=int, default=200, help='Max pills per prescription')
    
    parser.add_argument('--resume', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--workers', type=int, default=8, help='DataLoader workers')
    
    # DGX Spark specific
    parser.add_argument('--no-amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--cache-images', action='store_true', help='Cache images in memory')
    
    args = parser.parse_args()
    
    # Create config
    config = DGXSparkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        index_file=args.index_file,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        error_rate=args.error_rate,
        min_pills_per_prescription=args.min_pills,
        max_pills_per_prescription=args.max_pills,
        resume_from=args.resume,
        seed=args.seed,
        num_workers=args.workers,
        use_amp=not args.no_amp
    )
    
    # Build index if requested
    if args.build_index:
        index_path = os.path.join(args.data_dir, config.index_file)
        DatasetIndexer.build_index(args.data_dir, index_path)
        print(f"Index saved to {index_path}")
        return
    
    # Setup logging
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logging(config.output_dir)
    
    logger.info("=" * 70)
    logger.info("Prescription-Aware Pill Verification - DGX Spark")
    logger.info("=" * 70)
    
    # Setup device
    device = setup_dgx_spark_device(logger)
    amp_dtype = get_amp_dtype(config)
    logger.info(f"AMP dtype: {amp_dtype}")
    
    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    logger.info(f"Config: {config}")
    
    # Load index
    index_path = os.path.join(config.data_dir, config.index_file)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}\nRun with --build-index first")
    
    logger.info(f"Loading index from {index_path}")
    data_by_split, class_to_idx = DatasetIndexer.load_index(index_path)
    
    config.num_classes = len(class_to_idx)
    logger.info(f"Found {config.num_classes} NDC classes")
    
    # Analyze data
    train_prescriptions_raw = defaultdict(list)
    for item in data_by_split.get('train', []):
        train_prescriptions_raw[item['prescription_id']].append(item)
    
    if len(train_prescriptions_raw) == 0:
        raise ValueError("No training data found!")
    
    pill_counts = [len(pills) for pills in train_prescriptions_raw.values()]
    logger.info(f"Raw prescriptions: {len(train_prescriptions_raw)}")
    logger.info(f"Pills per Rx: min={min(pill_counts)}, max={max(pill_counts)}, mean={sum(pill_counts)/len(pill_counts):.1f}")
    
    filtered_count = sum(1 for pills in train_prescriptions_raw.values() 
                        if config.min_pills_per_prescription <= len(pills) <= config.max_pills_per_prescription)
    logger.info(f"After filter ({config.min_pills_per_prescription}-{config.max_pills_per_prescription}): {filtered_count}")
    
    if filtered_count == 0:
        raise ValueError(
            f"All prescriptions filtered out!\n"
            f"Suggested: --min-pills {min(pill_counts)} --max-pills {max(pill_counts)}"
        )
    
    # Create datasets
    train_dataset = PrescriptionDataset(
        data_dir=config.data_dir,
        split_data=data_by_split['train'],
        class_to_idx=class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=config.error_rate,
        is_training=True,
        cache_images=args.cache_images
    )
    
    val_dataset = PrescriptionDataset(
        data_dir=config.data_dir,
        split_data=data_by_split['valid'],
        class_to_idx=class_to_idx,
        max_pills=config.max_pills_per_prescription,
        min_pills=config.min_pills_per_prescription,
        error_rate=0.5,
        is_training=False,
        cache_images=args.cache_images
    )
    
    logger.info(f"Train prescriptions: {len(train_dataset)}")
    logger.info(f"Val prescriptions: {len(val_dataset)}")
    
    # Data loaders - optimized for DGX Spark unified memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=prescription_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=prescription_collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=True
    )
    
    # Create models
    embedding_model = EmbeddingExtractor(
        pretrained_path=args.resnet_checkpoint,
        num_classes=config.num_classes,
        freeze=True
    ).to(device)
    
    verifier = PrescriptionAwareVerifier(
        pill_embed_dim=embedding_model.embed_dim,
        ndc_vocab_size=config.num_classes,
        hidden_dim=256,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        use_gradient_checkpointing=config.use_gradient_checkpointing
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in verifier.parameters())
    trainable_params = sum(p.numel() for p in verifier.parameters() if p.requires_grad)
    logger.info(f"Verifier: {total_params:,} params, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        verifier.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Gradient scaler
    scaler = GradScaler(enabled=config.use_amp)
    
    # LR scheduler
    lr_scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.epochs,
        min_lr=config.min_lr
    )
    
    # Resume
    start_epoch = 0
    best_script_acc = 0.0
    
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        start_epoch, best_script_acc = load_checkpoint(
            config.resume_from,
            verifier, embedding_model, optimizer, scaler, device
        )
        start_epoch += 1
        logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_script_acc:.4f}")
    
    # Training loop
    logger.info("Starting training on DGX Spark...")
    
    for epoch in range(start_epoch, config.epochs):
        current_lr = lr_scheduler.step(epoch)
        logger.info(f"\nEpoch {epoch} | LR: {current_lr:.6f}")
        
        # Train
        train_metrics = train_epoch(
            verifier, embedding_model, train_loader,
            optimizer, scaler, config, device, epoch, logger, amp_dtype
        )
        
        logger.info(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"Script Acc: {train_metrics['script_acc']:.4f} | "
            f"Pill Acc: {train_metrics['pill_acc']:.4f}"
        )
        
        # Validate
        val_metrics = validate(
            verifier, embedding_model, val_loader,
            config, device, epoch, logger, amp_dtype
        )
        
        logger.info(
            f"Val   | Loss: {val_metrics['loss']:.4f} | "
            f"Script Acc: {val_metrics['script_acc']:.4f} | "
            f"Pill Acc: {val_metrics['pill_acc']:.4f}"
        )
        
        # Best model
        is_best = val_metrics['script_acc'] > best_script_acc
        if is_best:
            best_script_acc = val_metrics['script_acc']
            logger.info(f"New best: {best_script_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_every_n_epochs == 0 or is_best:
            save_checkpoint(
                verifier, embedding_model, optimizer, scaler,
                epoch, best_script_acc, config, config.output_dir, is_best
            )
        
        # Create collage
        if (epoch + 1) % config.collage_every_n_epochs == 0:
            collage_path = os.path.join(config.output_dir, f'collage_epoch_{epoch}.png')
            create_validation_collage(
                verifier, embedding_model, val_loader,
                device, collage_path, config.num_collage_samples, epoch, amp_dtype
            )
            logger.info(f"Saved collage: {collage_path}")
        
        # Garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\nTraining complete! Best script accuracy: {best_script_acc:.4f}")


if __name__ == '__main__':
    main()