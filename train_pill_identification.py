#!/usr/bin/env python3
"""
End-to-End Prescription-Aware Pill Verification (Corrupt-Padding Protected)
============================================================================
CRITICAL FIX APPLIED:
The ResNet backbone now actively filters out padding images before processing.
Previously, ~95% of inputs were black padding pixels, which destroyed
Batch Normalization statistics and prevented the model from learning visual features.
"""

import os
import sys
import re
import csv
import math
import random
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
from PIL import Image, ImageFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
from torchvision import transforms, models
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    data_dir: str = ""
    output_dir: str = "./output"
    index_file: str = "dataset_index.csv"
    
    # Architecture
    num_classes: int = 1228
    backbone: str = "resnet34"
    pill_embed_dim: int = 512
    verifier_hidden_dim: int = 256
    verifier_num_heads: int = 8
    verifier_num_layers: int = 4
    verifier_dropout: float = 0.1
    
    # Constraints
    max_pills_per_prescription: int = 200
    min_pills_per_prescription: int = 5
    
    # Training
    epochs: int = 100
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    # Differential Learning Rates
    backbone_lr: float = 5e-5
    classifier_lr: float = 1e-4
    verifier_lr: float = 1e-4
    weight_decay: float = 1e-2
    
    # Loss Weights
    script_loss_weight: float = 1.0
    pill_anomaly_loss_weight: float = 2.0  # Boosted to force visual learning
    pill_classification_loss_weight: float = 1.0
    
    error_rate: float = 0.5
    
    # Hardware
    use_amp: bool = True
    use_gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    
    num_workers: int = 4
    seed: int = 42
    resume_from: Optional[str] = None


# =============================================================================
# DDP & Logging
# =============================================================================

def setup_logging(rank, output_dir):
    logger = logging.getLogger("PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers = []
    if rank == 0:
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(os.path.join(output_dir, 'training.log'))
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        c_handler.setFormatter(fmt); f_handler.setFormatter(fmt)
        logger.addHandler(c_handler); logger.addHandler(f_handler)
    return logger

class DDPManager:
    def __init__(self, config):
        self.rank = int(os.environ.get('RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.is_distributed = self.world_size > 1
        self.device = torch.device(f'cuda:{self.local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    
    def setup(self):
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    
    def cleanup(self):
        if self.is_distributed: dist.destroy_process_group()
    
    @property
    def is_main(self): return self.rank == 0

@contextmanager
def ddp_sync_context(model, sync=True):
    if isinstance(model, DDP) and not sync:
        with model.no_sync(): yield
    else: yield


# =============================================================================
# Data Layer
# =============================================================================

class PrescriptionDataset(Dataset):
    def __init__(self, data_dir, split_data, class_to_idx, transform=None, 
                 max_pills=200, min_pills=5, error_rate=0.0, is_training=True):
        self.data_dir = Path(data_dir)
        self.split_data = split_data
        self.class_to_idx = class_to_idx
        self.transform = transform or self._get_transform(is_training)
        self.max_pills = max_pills
        self.min_pills = min_pills
        self.error_rate = error_rate
        self.is_training = is_training
        
        # Pre-group data
        grouped = defaultdict(list)
        for x in split_data: grouped[x['prescription_id']].append(x)
        self.prescriptions = {k: v for k, v in grouped.items() if min_pills <= len(v) <= max_pills}
        self.rx_ids = list(self.prescriptions.keys())
        
        # Build pill library for error injection
        self.library = defaultdict(list)
        if error_rate > 0:
            for rx in self.prescriptions.values():
                for p in rx: self.library[p['ndc']].append(p['relative_path'])

    def _get_transform(self, is_training):
        # ImageNet normalization
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if is_training:
            return transforms.Compose([
                transforms.Resize((672, 672)),
                transforms.RandomCrop(640),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1),
                transforms.ToTensor(),
                norm
            ])
        return transforms.Compose([
            transforms.Resize(640),
            transforms.CenterCrop(640),
            transforms.ToTensor(),
            norm
        ])

    def _load_img(self, path):
        try:
            return Image.open(self.data_dir / path).convert('RGB')
        except:
            return Image.new('RGB', (640, 640), (0, 0, 0))

    def __len__(self): return len(self.rx_ids)

    def __getitem__(self, idx):
        rx_id = self.rx_ids[idx]
        pills = self.prescriptions[rx_id]
        
        # Injection Logic
        wrong_indices = []
        is_script_correct = True
        
        if self.is_training and random.random() < self.error_rate:
            is_script_correct = False
            # Deep copy to avoid modifying dataset
            pills = [p.copy() for p in pills]
            n_mod = random.randint(1, max(1, len(pills)//3))
            wrong_indices = random.sample(range(len(pills)), n_mod)
            
            for i in wrong_indices:
                orig_ndc = pills[i]['ndc']
                candidates = [k for k in self.library.keys() if k != orig_ndc]
                if candidates:
                    new_ndc = random.choice(candidates)
                    pills[i]['ndc'] = new_ndc
                    pills[i]['relative_path'] = random.choice(self.library[new_ndc])

        # Load Images
        images = torch.stack([self.transform(self._load_img(p['relative_path'])) for p in pills])
        labels = torch.tensor([self.class_to_idx[p['ndc']] for p in pills], dtype=torch.long)
        
        # Expected NDCs (Based on original prescription)
        orig_pills = self.prescriptions[rx_id]
        expected_ndcs = list(set(self.class_to_idx[p['ndc']] for p in orig_pills))
        expected_counts = [sum(1 for p in orig_pills if p['ndc'] == k) for k in [p['ndc'] for p in orig_pills]]
        # Simplify count logic: just count occurrences of unique NDCs
        uniq_map = defaultdict(int)
        for p in orig_pills: uniq_map[self.class_to_idx[p['ndc']]] += 1
        exp_ndc_tensor = torch.tensor(list(uniq_map.keys()), dtype=torch.long)
        exp_cnt_tensor = torch.tensor(list(uniq_map.values()), dtype=torch.float)

        pill_correct = torch.ones(len(pills))
        pill_correct[wrong_indices] = 0.0

        return {
            'images': images,
            'labels': labels,
            'expected_ndcs': exp_ndc_tensor,
            'expected_counts': exp_cnt_tensor,
            'is_correct': torch.tensor(float(is_script_correct)),
            'pill_correct': pill_correct,
            'num_pills': len(pills)
        }

def collate_fn(batch):
    B = len(batch)
    max_p = max(x['num_pills'] for x in batch)
    max_e = max(len(x['expected_ndcs']) for x in batch)
    C, H, W = batch[0]['images'].shape[1:]
    
    out = {
        'images': torch.zeros(B, max_p, C, H, W),
        'labels': torch.full((B, max_p), -100, dtype=torch.long),
        'pill_mask': torch.ones(B, max_p, dtype=torch.bool), # True=Pad
        'pill_correct': torch.zeros(B, max_p),
        
        'expected_ndcs': torch.zeros(B, max_e, dtype=torch.long),
        'expected_counts': torch.zeros(B, max_e, dtype=torch.float),
        'expected_mask': torch.ones(B, max_e, dtype=torch.bool), # True=Pad
        
        'is_correct': torch.tensor([x['is_correct'] for x in batch])
    }
    
    for i, b in enumerate(batch):
        n, ne = b['num_pills'], len(b['expected_ndcs'])
        out['images'][i, :n] = b['images']
        out['labels'][i, :n] = b['labels']
        out['pill_mask'][i, :n] = False
        out['pill_correct'][i, :n] = b['pill_correct']
        
        out['expected_ndcs'][i, :ne] = b['expected_ndcs']
        out['expected_counts'][i, :ne] = b['expected_counts']
        out['expected_mask'][i, :ne] = False
        
    return out


# =============================================================================
# Models - FIXED FOR PADDING CORRUPTION
# =============================================================================

class ResNetBackbone(nn.Module):
    def __init__(self, name, num_classes, checkpointing):
        super().__init__()
        self.use_chk = checkpointing
        weights = models.ResNet34_Weights.DEFAULT if name=='resnet34' else models.ResNet50_Weights.DEFAULT
        base = models.resnet34(weights=weights) if name=='resnet34' else models.resnet50(weights=weights)
        self.dim = 512 if name=='resnet34' else 2048
        
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layers = nn.ModuleList([base.layer1, base.layer2, base.layer3, base.layer4])
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(self.dim, num_classes)

    def forward(self, x):
        # x shape: (N_valid, C, H, W) - ALREADY FILTERED, NO PADDING
        if self.use_chk and self.training:
            x = checkpoint(self.stem, x, use_reentrant=False)
            for layer in self.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = self.stem(x)
            for layer in self.layers: x = layer(x)
            
        x = self.avgpool(x).flatten(1)
        logits = self.classifier(x)
        return x, logits

class PrescriptionVerifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.verifier_hidden_dim
        self.ndc_emb = nn.Embedding(cfg.num_classes, d)
        
        # Encoders
        self.rx_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d*4, 0.1, batch_first=True), 2)
        self.pill_enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, d*4, 0.1, batch_first=True), cfg.verifier_num_layers)
        
        # Projection
        self.proj = nn.Linear(cfg.pill_embed_dim + 1, d)
        
        # Cross Attention
        self.cross = nn.MultiheadAttention(d, 8, batch_first=True)
        self.norm = nn.LayerNorm(d)
        
        # Heads
        self.pool_tok = nn.Parameter(torch.randn(1, 1, d))
        self.head_script = nn.Linear(d, 1)
        self.head_anom = nn.Linear(d, 1)

    def forward(self, pill_emb, pill_conf, rx_ndc, rx_cnt, p_mask, rx_mask):
        # 1. Encode Script
        rx = self.ndc_emb(rx_ndc) + (rx_cnt.unsqueeze(-1) * 0.05)
        rx = self.rx_enc(rx, src_key_padding_mask=rx_mask)
        
        # 2. Pill Features
        x = self.proj(torch.cat([pill_emb, pill_conf], -1))
        
        # 3. Cross Attn (Pills attend to Script)
        # Note: key_padding_mask for cross attention applies to Key/Value (the Script)
        attn, _ = self.cross(x, rx, rx, key_padding_mask=rx_mask)
        x = self.norm(x + attn)
        
        # 4. Self Attn
        x = self.pill_enc(x, src_key_padding_mask=p_mask)
        
        # 5. Anomaly
        anom = self.head_anom(x).squeeze(-1)
        
        # 6. Script Level
        B = x.size(0)
        tok = self.pool_tok.expand(B, -1, -1)
        # Concatenate token + pills for pooling
        combined = torch.cat([tok, x], dim=1)
        # Extend mask for token (False = valid)
        tok_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        combined_mask = torch.cat([tok_mask, p_mask], dim=1)
        
        # Simple Mean Pooling of valid tokens
        # (Transformer pooling is better but mean is robust for now)
        mask_float = (~combined_mask).float().unsqueeze(-1)
        pooled = (combined * mask_float).sum(1) / (mask_float.sum(1) + 1e-8)
        
        script = self.head_script(pooled)
        return script, anom

class EndToEndModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNetBackbone(cfg.backbone, cfg.num_classes, cfg.use_gradient_checkpointing)
        self.verifier = PrescriptionVerifier(cfg)
        self.embed_dim = cfg.pill_embed_dim
        self.num_classes = cfg.num_classes

    def forward(self, images, rx_ndc, rx_cnt, p_mask, rx_mask):
        """
        Args:
            images: (B, N, C, H, W) - Contains PADDING (Zeros)
            p_mask: (B, N) - True where padding exists
        """
        B, N, C, H, W = images.shape
        
        # --- CRITICAL FIX START ---
        # 1. Flatten to (B*N, ...)
        flat_imgs = images.view(-1, C, H, W)
        flat_mask = p_mask.view(-1)
        
        # 2. Filter: Select ONLY valid images
        # We perform boolean indexing to remove padding images entirely
        valid_indices = torch.nonzero(~flat_mask).squeeze()
        valid_imgs = flat_imgs[valid_indices]
        
        # 3. Forward Pass (Only on valid images)
        # This ensures Batch Norm sees only real data
        valid_emb, valid_logits = self.backbone(valid_imgs)
        
        # 4. Scatter Back: Reconstruct (B, N) structure
        # Initialize containers with zeros
        emb_full = torch.zeros(B * N, self.embed_dim, device=images.device, dtype=valid_emb.dtype)
        logits_full = torch.zeros(B * N, self.num_classes, device=images.device, dtype=valid_logits.dtype)
        
        # Place valid results back into their original positions
        emb_full.index_copy_(0, valid_indices, valid_emb)
        logits_full.index_copy_(0, valid_indices, valid_logits)
        
        # Reshape to (B, N, ...)
        emb_out = emb_full.view(B, N, -1)
        logits_out = logits_full.view(B, N, -1)
        # --- CRITICAL FIX END ---
        
        # Calculate Confidence
        probs = F.softmax(logits_out, dim=-1)
        confs = probs.max(dim=-1).values.unsqueeze(-1)
        
        # Verifier
        s_logit, a_logits = self.verifier(emb_out, confs, rx_ndc, rx_cnt, p_mask, rx_mask)
        
        return s_logit, a_logits, logits_out

# =============================================================================
# Training Loop
# =============================================================================

def train_one_epoch(model, loader, opt, scaler, ddp, cfg):
    model.train()
    meters = defaultdict(float)
    counts = defaultdict(int)
    
    if ddp.is_main: pbar = tqdm(loader)
    
    for i, batch in enumerate(loader):
        batch = {k: v.to(ddp.device, non_blocking=True) for k,v in batch.items()}
        
        with ddp_sync_context(model, (i+1)%cfg.gradient_accumulation_steps==0):
            with autocast(enabled=cfg.use_amp):
                s_l, a_l, p_l = model(
                    batch['images'], batch['expected_ndcs'], batch['expected_counts'],
                    batch['pill_mask'], batch['expected_mask']
                )
                
                # Losses
                loss_s = F.binary_cross_entropy_with_logits(s_l.squeeze(-1), batch['is_correct'])
                
                valid = ~batch['pill_mask']
                loss_a = 0
                if valid.any():
                    l_a_raw = F.binary_cross_entropy_with_logits(a_l, 1.0-batch['pill_correct'], reduction='none')
                    loss_a = (l_a_raw * valid).sum() / (valid.sum() + 1e-6)
                
                # ResNet Classification Loss (Flattened)
                loss_c = F.cross_entropy(p_l.view(-1, cfg.num_classes), batch['labels'].view(-1), ignore_index=-100)
                
                loss = (loss_s * cfg.script_loss_weight + 
                        loss_a * cfg.pill_anomaly_loss_weight + 
                        loss_c * cfg.pill_classification_loss_weight)
                
                loss = loss / cfg.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        if (i+1)%cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt); scaler.update(); opt.zero_grad()
        
        # Metrics
        bs = batch['images'].size(0)
        with torch.no_grad():
            acc_s = ((torch.sigmoid(s_l.squeeze())>0.5) == batch['is_correct']).float().mean()
            acc_p = 0
            if valid.any():
                preds = p_l.argmax(-1)
                targs = batch['labels']
                mask = (targs != -100)
                acc_p = (preds[mask] == targs[mask]).float().mean()
        
        for k,v in {'loss': loss.item()*cfg.gradient_accumulation_steps, 'acc_s': acc_s.item(), 'acc_p': acc_p.item()}.items():
            meters[k] += v * bs
            counts[k] += bs
            
        if ddp.is_main: pbar.set_postfix({'loss': meters['loss']/counts['loss'], 'acc_p': meters['acc_p']/counts['acc_p']})

    return {k: v/counts[k] for k,v in meters.items()}

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--build-index', action='store_true')
    args = parser.parse_args()
    
    # Simple CSV Indexer
    if args.build_index:
        print("Indexing...")
        with open(os.path.join(args.data_dir, 'dataset_index.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(['split','ndc','prescription_id','patch_no','relative_path'])
            for split in ['train', 'valid']:
                p = Path(args.data_dir)/split
                if not p.exists(): continue
                for d in p.iterdir():
                    if not d.is_dir(): continue
                    for img in d.glob('*.jpg'):
                        # filename fmt: NDC_RxID_patch_N.jpg
                        parts = img.stem.split('_') 
                        if len(parts) >= 4:
                            w.writerow([split, d.name, parts[1], parts[3], str(img.relative_to(args.data_dir))])
        return

    cfg = TrainingConfig(data_dir=args.data_dir)
    ddp = DDPManager(cfg); ddp.setup()
    logger = setup_logging(ddp.rank, cfg.output_dir)
    
    # Load Index
    rows = []
    with open(os.path.join(cfg.data_dir, cfg.index_file)) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Map Classes
    ndcs = sorted(list(set(r['ndc'] for r in rows)))
    cls_map = {n: i for i, n in enumerate(ndcs)}
    cfg.num_classes = len(ndcs)
    
    train_data = [r for r in rows if r['split']=='train']
    valid_data = [r for r in rows if r['split']=='valid']
    
    ds_train = PrescriptionDataset(cfg.data_dir, train_data, cls_map, is_training=True)
    ds_valid = PrescriptionDataset(cfg.data_dir, valid_data, cls_map, is_training=False)
    
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, 
                          sampler=DistributedSampler(ds_train) if ddp.is_distributed else None,
                          collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)
                          
    # Model Setup
    model = EndToEndModel(cfg).to(ddp.device)
    if ddp.is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[ddp.local_rank], find_unused_parameters=False)
        
    opt = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if 'backbone' in n], 'lr': cfg.backbone_lr},
        {'params': [p for n,p in model.named_parameters() if 'backbone' not in n], 'lr': cfg.verifier_lr}
    ], weight_decay=cfg.weight_decay)
    
    scaler = GradScaler(enabled=cfg.use_amp)

    # Train
    for ep in range(cfg.epochs):
        if ddp.is_distributed: dl_train.sampler.set_epoch(ep)
        res = train_one_epoch(model, dl_train, opt, scaler, ddp, cfg)
        if ddp.is_main:
            logger.info(f"Ep {ep}: Loss={res['loss']:.4f} | ScriptAcc={res['acc_s']:.4f} | PillAcc={res['acc_p']:.4f}")
            if (ep+1)%5==0: 
                torch.save(model.state_dict(), f"{cfg.output_dir}/model_{ep}.pt")

    ddp.cleanup()

if __name__ == '__main__':
    main()