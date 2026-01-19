import torch
import torch.nn as nn
from torch.utils.data import Dataset
from openvino.runtime import Core
import numpy as np
import os
import glob
from PIL import Image

# --- 1. DATASET ---
class PrescriptionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        # Sort classes to ensure consistent indexing across runs
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"[Dataset] Indexing {root_dir}...")
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            grouped_patches = {} 
            
            # Group patches by Prescription ID (filename prefix)
            for img_path in glob.glob(os.path.join(cls_folder, "*.jpg")):
                try:
                    filename = os.path.basename(img_path)
                    # Logic: "Amox_Rx001_patch_0.jpg" -> PID: "Amox_Rx001"
                    pid = filename.split('_patch_')[0]
                    if pid not in grouped_patches: grouped_patches[pid] = []
                    grouped_patches[pid].append(img_path)
                except: continue
            
            for pid, paths in grouped_patches.items():
                self.samples.append((paths, class_idx, pid))
        
        print(f"[Dataset] Found {len(self.samples)} prescriptions across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_paths, label, pid = self.samples[idx]
        images = []
        for p in patch_paths:
            try:
                with Image.open(p) as img_raw:
                    img = img_raw.convert('RGB')
                    if self.transform: img = self.transform(img)
                    images.append(img)
            except: continue
        
        if len(images) == 0:
            return torch.zeros((1, 3, 224, 224)), label, pid
            
        return torch.stack(images), label, pid

def collate_mil_pad(batch):
    batch_images, labels, pids = zip(*batch)
    MAX_PILLS = 64  # Cap per bag to prevent RAM spikes
    padded_batch = []
    mask = [] 
    
    for img_stack in batch_images:
        if img_stack.shape[0] > MAX_PILLS:
            perm = torch.randperm(img_stack.shape[0])[:MAX_PILLS]
            img_stack = img_stack[perm]
        padded_batch.append(img_stack)
        
    max_pills_batch = max([x.shape[0] for x in padded_batch])
    c, h, w = padded_batch[0].shape[1:]
    
    final_batch = []
    final_mask = []
    
    for stack in padded_batch:
        n = stack.shape[0]
        pad_size = max_pills_batch - n
        m = torch.cat([torch.ones(n), torch.zeros(pad_size)])
        final_mask.append(m)
        if pad_size > 0:
            padding = torch.zeros((pad_size, c, h, w))
            final_batch.append(torch.cat([stack, padding], dim=0))
        else:
            final_batch.append(stack)
            
    return torch.stack(final_batch), torch.tensor(labels), torch.stack(final_mask), pids

# --- 2. OPENVINO BACKBONE WRAPPER ---
class OpenVINOResNet34(nn.Module):
    def __init__(self, xml_path, bin_path):
        super().__init__()
        self.ie = Core()
        
        print(f"[Backbone] Loading OpenVINO IR: {os.path.basename(xml_path)}")
        model = self.ie.read_model(model=xml_path, weights=bin_path)
        
        # Compile for CPU (Standard for OpenVINO)
        self.compiled_model = self.ie.compile_model(model=model, device_name="CPU")
        self.output_layer = self.compiled_model.output(0)
        
        # Determine Output Size
        out_shape = self.output_layer.shape
        # Typically [1, 512, 1, 1] for ResNet34 features or [1, 1000] for logits
        self.feature_dim = out_shape[1] 
        print(f"[Backbone] Loaded. Feature Dimension: {self.feature_dim}")
        
        if self.feature_dim > 2000 and self.feature_dim != 2048:
             print("[WARNING] Output dim looks like class logits, not features!")
             print("          For best results, export model without the final FC layer.")

    def forward(self, x):
        # Input: PyTorch Tensor [B, C, H, W] on CPU or GPU
        # OpenVINO requires: Numpy [B, C, H, W] on CPU
        
        if x.device.type == 'cuda':
            x_np = x.cpu().numpy()
        else:
            x_np = x.numpy()
            
        # Inference
        results = self.compiled_model([x_np])[self.output_layer]
        
        # Convert back to Torch
        features = torch.from_numpy(results)
        
        # Flatten if [B, 512, 1, 1] -> [B, 512]
        if len(features.shape) == 4:
            features = features.view(features.size(0), -1)
            
        return features

# --- 3. HYBRID MIL MODEL ---
class GatedAttentionMIL(nn.Module):
    def __init__(self, num_classes, xml_path, bin_path):
        super(GatedAttentionMIL, self).__init__()
        
        # Frozen Backbone
        self.backbone = OpenVINOResNet34(xml_path, bin_path)
        self.feature_dim = self.backbone.feature_dim
        
        # Trainable MIL Head
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        self.bag_classifier = nn.Linear(self.feature_dim, num_classes)
        self.instance_classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, mask=None):
        # x: [Batch, Pills, C, H, W]
        B, M, C, H, W = x.shape
        
        # 1. Feature Extraction (CPU / OpenVINO)
        # We flatten to [B*M, C, H, W] for the backbone
        flat_x = x.view(B * M, C, H, W)
        
        with torch.no_grad():
            features = self.backbone(flat_x) # Output: [B*M, 512]
        
        # 2. Move Features to GPU for Training
        if torch.cuda.is_available():
            features = features.cuda()
            if mask is not None: mask = mask.cuda()
            
        features = features.view(B, M, -1)
        
        # 3. MIL Attention (GPU / PyTorch)
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U) 
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            A = A.masked_fill(mask_expanded == 0, -1e9)
        
        A = torch.softmax(A, dim=1) 
        bag_embedding = torch.sum(features * A, dim=1) 
        
        bag_logits = self.bag_classifier(bag_embedding)
        instance_logits = self.instance_classifier(features)
        
        return bag_logits, instance_logits, A
