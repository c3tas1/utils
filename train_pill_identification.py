#!/usr/bin/env python3
"""
End-to-End Prescription Verification (Curriculum Learning + Padding Protection)
================================================================================
Target Hardware: NVIDIA DGX A100 (8x A100 40GB GPUs)

CORE LOGIC:
1. PADDING PROTECTION: Padding images are filtered out before the Backbone.
   This prevents Batch Norm corruption (essential for small batches).

2. CURRICULUM LEARNING:
   - Phase 1: Train ONLY the Backbone (ResNet) using classification loss.
     Verifier loss weights are set to 0.0.
   - Phase 2: Once val_pill_acc > 15%, enable Verifier (Script + Anomaly losses).
     This prevents the "gradient starvation" issue where the model ignores images.

Usage:
    torchrun --nproc_per_node=8 train_e2e_final.py --data-dir /path/to/data --build-index
    torchrun --nproc_per_node=8 train_e2e_final.py --data-dir /path/to/data --epochs 100
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
    
    # Constraints
    max_pills: int = 200
    min_pills: int = 5
    
    # Training
    epochs: int = 100
    batch_size: int = 2
    grad_accum_steps: int = 4
    
    # LRs
    backbone_lr: float = 5e-5
    verifier_lr: float = 1e-4
    weight_decay: float = 1e-2
    
    # Curriculum Threshold
    curriculum_threshold: float = 0.15  # 15% Accuracy triggers Phase 2
    
    # Hardware
    use_chk: bool = True  # Gradient Checkpointing (Required for 640x640)
    
    num_workers: int = 4
    resume_from: Optional[str] = None


# =============================================================================
# Infrastructure
# =============================================================================

def setup_logger(rank, output_dir):
    logger = logging.getLogger("PillVerifier")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
        
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(fmt)
        logger.addHandler(c_handler)
        
        f_handler = logging.FileHandler(os.path.join(output_dir, 'train.log'))
        f_handler.setFormatter(fmt)
        logger.addHandler(f_handler)
    return logger

class DDPManager:
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_dist = self.world_size > 1
        self.device = torch.device(f'cuda:{self.local_rank}')
    
    def setup(self):
        if self.is_dist:
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    
    def cleanup(self):
        if self.is_dist: dist.destroy_process_group()
        
    @property
    def is_main(self): return self.rank == 0

# =============================================================================
# Data
# =============================================================================

class PrescriptionDataset(Dataset):
    def __init__(self, data_dir, index_data, class_map, is_training=True):
        self.data_dir = Path(data_dir)
        self.cls_map = class_map
        self.is_training = is_training
        
        # 640x640 Transforms
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if is_training:
            self.tx = transforms.Compose([
                transforms.Resize((672, 672)),
                transforms.RandomCrop(640),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.1),
                transforms.ToTensor(), norm
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize(640),
                transforms.CenterCrop(640),
                transforms.ToTensor(), norm
            ])
            
        # Grouping
        grouped = defaultdict(list)
        for r in index_data: grouped[r['prescription_id']].append(r)
        
        self.rxs = []
        for rx_id, pills in grouped.items():
            if 5 <= len(pills) <= 200:
                self.rxs.append(sorted(pills, key=lambda x: int(x['patch_no'])))
                
        # Library for Error Injection
        self.lib = defaultdict(list)
        if is_training:
            for rx in self.rxs:
                for p in rx: self.lib[p['ndc']].append(p['relative_path'])

    def __len__(self): return len(self.rxs)

    def __getitem__(self, idx):
        pills = self.rxs[idx]
        is_correct = True
        wrong_mask = torch.ones(len(pills)) # 1 = Correct
        
        # Injection (50% chance during training)
        if self.is_training and random.random() < 0.5:
            is_correct = False
            pills = [p.copy() for p in pills] # Deep copy
            n_err = random.randint(1, max(1, len(pills)//3))
            err_idx = random.sample(range(len(pills)), n_err)
            
            for i in err_idx:
                wrong_mask[i] = 0.0
                orig = pills[i]['ndc']
                # Pick random DIFFERENT ndc
                opts = [k for k in self.lib.keys() if k != orig]
                if opts:
                    new_ndc = random.choice(opts)
                    pills[i]['ndc'] = new_ndc
                    pills[i]['relative_path'] = random.choice(self.lib[new_ndc])

        # Load Images
        imgs = []
        lbls = []
        for p in pills:
            try:
                img = Image.open(self.data_dir / p['relative_path']).convert('RGB')
                imgs.append(self.tx(img))
            except:
                imgs.append(torch.zeros(3, 640, 640)) # Safe fallback
            lbls.append(self.cls_map[p['ndc']])

        # Expected NDCs (from original prescription logic - implicit here)
        # Simplified: Just passing unique NDCs present in the *original* list would be ideal
        # But for training speed, we can just use the labels as "expected" if correct
        # Ideally, you pass the text description of the RX separately.
        # For this script, we assume 'labels' represents the visual ground truth.
        
        return {
            'images': torch.stack(imgs),
            'labels': torch.tensor(lbls, dtype=torch.long),
            'is_correct': torch.tensor(float(is_correct)),
            'pill_correct': wrong_mask,
            'num_pills': len(pills)
        }

def collate_fn(batch):
    B = len(batch)
    max_p = max(b['num_pills'] for b in batch)
    C, H, W = batch[0]['images'].shape[1:]
    
    out = {
        'images': torch.zeros(B, max_p, C, H, W),
        'labels': torch.full((B, max_p), -100, dtype=torch.long),
        'mask': torch.ones(B, max_p, dtype=torch.bool), # True=Pad
        'pill_correct': torch.zeros(B, max_p),
        'is_correct': torch.tensor([b['is_correct'] for b in batch])
    }
    
    for i, b in enumerate(batch):
        n = b['num_pills']
        out['images'][i, :n] = b['images']
        out['labels'][i, :n] = b['labels']
        out['mask'][i, :n] = False
        out['pill_correct'][i, :n] = b['pill_correct']
        
    return out

# =============================================================================
# Model (Padding Protected)
# =============================================================================

class ResNetBackbone(nn.Module):
    def __init__(self, num_classes, use_chk):
        super().__init__()
        self.use_chk = use_chk
        m = models.resnet34(weights='IMAGENET1K_V1')
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layers = nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])
        self.avgpool = m.avgpool
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x is (N_valid, 3, 640, 640) -> NO PADDING HERE
        if self.use_chk and self.training:
            x = checkpoint(self.stem, x, use_reentrant=False)
            for l in self.layers: x = checkpoint(l, x, use_reentrant=False)
        else:
            x = self.stem(x)
            for l in self.layers: x = l(x)
            
        feat = self.avgpool(x).flatten(1)
        logits = self.classifier(feat)
        return feat, logits

class Verifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.verifier_hidden_dim
        self.proj = nn.Linear(cfg.pill_embed_dim + 1, d)
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, 8, d*4, 0.1, batch_first=True, norm_first=True), 
            num_layers=4
        )
        self.head_s = nn.Linear(d, 1)
        self.head_a = nn.Linear(d, 1)
        self.tok = nn.Parameter(torch.randn(1, 1, d))

    def forward(self, emb, conf, mask):
        # emb: (B, N, 512)
        x = self.proj(torch.cat([emb, conf], -1)) # (B, N, 256)
        
        # Add Token
        B = x.size(0)
        tok = self.tok.expand(B, -1, -1)
        x = torch.cat([tok, x], dim=1) # (B, N+1, 256)
        
        # Extend Mask (Token is valid)
        # mask is True for Padding. Add False for token.
        tok_mask = torch.zeros(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([tok_mask, mask], dim=1)
        
        feat = self.enc(x, src_key_padding_mask=full_mask)
        
        # Script prediction from Token [0]
        script_logits = self.head_s(feat[:, 0, :])
        
        # Anomaly prediction from Pills [1:]
        anom_logits = self.head_a(feat[:, 1:, :]).squeeze(-1)
        
        return script_logits, anom_logits

class EndToEndModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNetBackbone(cfg.num_classes, cfg.use_chk)
        self.verifier = Verifier(cfg)
        self.dim = 512
        self.n_cls = cfg.num_classes

    def forward(self, images, mask):
        """
        PAD PROTECTION LOGIC:
        1. Flatten Batch
        2. Remove Padding Indices
        3. Run Backbone
        4. Scatter results back to (B, N)
        """
        B, N, C, H, W = images.shape
        
        # 1. Flatten
        flat_img = images.view(-1, C, H, W)
        flat_msk = mask.view(-1)
        
        # 2. Filter
        valid_idx = torch.nonzero(~flat_msk).squeeze()
        valid_img = flat_img[valid_idx]
        
        # 3. Backbone (Only Real Images)
        valid_feat, valid_logits = self.backbone(valid_img)
        
        # 4. Scatter Back
        # Create empty buffers (B*N)
        full_feat = torch.zeros(B*N, self.dim, device=images.device, dtype=valid_feat.dtype)
        full_logits = torch.zeros(B*N, self.n_cls, device=images.device, dtype=valid_logits.dtype)
        
        # Scatter
        full_feat.index_copy_(0, valid_idx, valid_feat)
        full_logits.index_copy_(0, valid_idx, valid_logits)
        
        # Reshape
        feat_out = full_feat.view(B, N, -1)
        logits_out = full_logits.view(B, N, -1)
        
        # 5. Verifier
        # Calc confidence
        conf = F.softmax(logits_out, dim=-1).max(dim=-1).values.unsqueeze(-1)
        
        s_logit, a_logits = self.verifier(feat_out, conf, mask)
        
        return s_logit, a_logits, logits_out

# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, opt, scaler, ddp, cfg, phase, logger):
    model.train()
    meters = defaultdict(float)
    count = 0
    
    if ddp.is_main: 
        pbar = tqdm(loader, desc=f"Phase: {phase}")
    
    for i, batch in enumerate(loader):
        imgs = batch['images'].to(ddp.device, non_blocking=True)
        lbls = batch['labels'].to(ddp.device, non_blocking=True)
        mask = batch['mask'].to(ddp.device, non_blocking=True)
        is_corr = batch['is_correct'].to(ddp.device, non_blocking=True)
        p_corr = batch['pill_correct'].to(ddp.device, non_blocking=True)
        
        with autocast(enabled=True):
            s_l, a_l, p_l = model(imgs, mask)
            
            # 1. Classification Loss (Always Active)
            # Flatten to ignore padding (-100 handled by CE)
            loss_cls = F.cross_entropy(p_l.view(-1, cfg.num_classes), lbls.view(-1), ignore_index=-100)
            
            # 2. Verifier Losses (Curriculum Controlled)
            loss_s = F.binary_cross_entropy_with_logits(s_l.squeeze(-1), is_corr)
            
            # Anomaly loss (masked)
            valid = ~mask
            loss_a = torch.tensor(0.0, device=ddp.device)
            if valid.any():
                l_raw = F.binary_cross_entropy_with_logits(a_l, 1.0 - p_corr, reduction='none')
                loss_a = (l_raw * valid).sum() / (valid.sum() + 1e-6)
            
            # Combine based on Phase
            if phase == "BACKBONE_ONLY":
                # Verifier is effectively detached. 
                # We weight these 0.0 so gradients don't flow to Verifier,
                # BUT we calculate them so DDP (find_unused_parameters=True) doesn't hang.
                loss = loss_cls
                # Optional: loss += 0.0 * (loss_s + loss_a) to keep graph connected if needed,
                # but find_unused_parameters=True handles disconnection.
            else:
                loss = loss_cls + loss_s + (loss_a * 2.0) # Boost anomaly detection
            
            loss = loss / cfg.grad_accum_steps
            
        scaler.scale(loss).backward()
        
        if (i+1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
        # Metrics
        bs = imgs.size(0)
        with torch.no_grad():
            acc_s = ((torch.sigmoid(s_l.squeeze()) > 0.5) == is_corr).float().mean()
            
            # Pill Acc
            preds = p_l.argmax(-1)
            valid_mask = (lbls != -100)
            acc_p = torch.tensor(0.0, device=ddp.device)
            if valid_mask.any():
                acc_p = (preds[valid_mask] == lbls[valid_mask]).float().mean()
        
        meters['loss'] += loss.item() * cfg.grad_accum_steps * bs
        meters['acc_s'] += acc_s.item() * bs
        meters['acc_p'] += acc_p.item() * bs
        count += bs
        
        if ddp.is_main:
            pbar.set_postfix({'acc_p': meters['acc_p']/count, 'loss': meters['loss']/count})
            
    return {k: v/count for k,v in meters.items()}

@torch.no_grad()
def validate(model, loader, ddp):
    model.eval()
    meters = defaultdict(float)
    count = 0
    
    for batch in loader:
        imgs = batch['images'].to(ddp.device)
        lbls = batch['labels'].to(ddp.device)
        mask = batch['mask'].to(ddp.device)
        
        with autocast(enabled=True):
            _, _, p_l = model(imgs, mask)
        
        preds = p_l.argmax(-1)
        valid = (lbls != -100)
        
        bs = imgs.size(0)
        acc_p = 0.0
        if valid.any():
            acc_p = (preds[valid] == lbls[valid]).float().mean().item()
            
        meters['acc_p'] += acc_p * bs
        count += bs
        
    # Sync
    total_acc = torch.tensor(meters['acc_p'], device=ddp.device)
    total_cnt = torch.tensor(count, device=ddp.device)
    dist.all_reduce(total_acc)
    dist.all_reduce(total_cnt)
    
    return total_acc.item() / total_cnt.item()

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--build-index', action='store_true')
    args = parser.parse_args()
    
    # 1. Indexing
    if args.build_index:
        print("Building Index...")
        with open(os.path.join(args.data_dir, 'dataset_index.csv'), 'w') as f:
            w = csv.writer(f)
            w.writerow(['split','ndc','prescription_id','patch_no','relative_path'])
            for split in ['train', 'valid']:
                s_dir = Path(args.data_dir) / split
                if not s_dir.exists(): continue
                for d in s_dir.iterdir():
                    if d.is_dir():
                        for img in d.glob('*.jpg'):
                            # Expected fmt: NDC_RxID_patch_N.jpg
                            parts = img.stem.split('_')
                            if len(parts) >= 4:
                                w.writerow([split, d.name, parts[1], parts[3], str(img.relative_to(args.data_dir))])
        return

    # 2. Config & Setup
    cfg = TrainingConfig(data_dir=args.data_dir)
    ddp = DDPManager()
    ddp.setup()
    logger = setup_logger(ddp.rank, cfg.output_dir)
    
    # 3. Data Loading
    idx_path = os.path.join(cfg.data_dir, cfg.index_file)
    with open(idx_path) as f: rows = list(csv.DictReader(f))
    
    ndcs = sorted(list(set(r['ndc'] for r in rows)))
    cls_map = {n: i for i, n in enumerate(ndcs)}
    cfg.num_classes = len(ndcs)
    
    if ddp.is_main: logger.info(f"Loaded {len(rows)} samples, {cfg.num_classes} classes.")
    
    ds_train = PrescriptionDataset(cfg.data_dir, [r for r in rows if r['split']=='train'], cls_map, True)
    ds_valid = PrescriptionDataset(cfg.data_dir, [r for r in rows if r['split']=='valid'], cls_map, False)
    
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, sampler=DistributedSampler(ds_train), 
                          num_workers=4, pin_memory=True, collate_fn=collate_fn, drop_last=True)
    dl_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, sampler=DistributedSampler(ds_valid, shuffle=False),
                          num_workers=4, pin_memory=True, collate_fn=collate_fn)
                          
    # 4. Model
    model = EndToEndModel(cfg).to(ddp.device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # Critical for BS=2
    # find_unused_parameters=True is REQUIRED because in Phase 1 we ignore Verifier
    model = DDP(model, device_ids=[ddp.local_rank], find_unused_parameters=True)
    
    opt = torch.optim.AdamW([
        {'params': model.module.backbone.parameters(), 'lr': cfg.backbone_lr},
        {'params': model.module.verifier.parameters(), 'lr': cfg.verifier_lr}
    ], weight_decay=cfg.weight_decay)
    
    scaler = GradScaler()
    
    # 5. Training Loop
    phase = "BACKBONE_ONLY"
    
    for ep in range(cfg.epochs):
        dl_train.sampler.set_epoch(ep)
        
        # Train
        metrics = train_epoch(model, dl_train, opt, scaler, ddp, cfg, phase, logger)
        
        # Validate (Pill Accuracy)
        val_acc = validate(model, dl_valid, ddp)
        
        if ddp.is_main:
            logger.info(f"Ep {ep} [{phase}]: Train Loss={metrics['loss']:.4f} Pill Acc={metrics['acc_p']:.4f} | Val Acc={val_acc:.4f}")
            
            # Checkpoint
            if (ep+1) % 5 == 0:
                torch.save(model.state_dict(), f"{cfg.output_dir}/model_{ep}.pt")
                
        # Curriculum Transition Logic
        # Sync phase decision across all GPUs based on Val Acc
        acc_tensor = torch.tensor(val_acc, device=ddp.device)
        dist.broadcast(acc_tensor, src=0)
        
        if phase == "BACKBONE_ONLY" and acc_tensor.item() > cfg.curriculum_threshold:
            phase = "END_TO_END"
            if ddp.is_main: logger.info(f">>> SWITCHING TO END-TO-END TRAINING (Val Acc > {cfg.curriculum_threshold}) <<<")

    ddp.cleanup()

if __name__ == '__main__':
    main()