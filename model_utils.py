import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image

# --- 1. Helper: Un-Normalization ---
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

# --- 2. Custom Dataset ---
class PrescriptionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"Indexing dataset from {root_dir}...")
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            
            grouped_patches = {} 
            for img_path in glob.glob(os.path.join(cls_folder, "*.jpg")):
                filename = os.path.basename(img_path)
                try:
                    parts = filename.split('_patch_')
                    prescription_id = parts[0] 
                    if prescription_id not in grouped_patches:
                        grouped_patches[prescription_id] = []
                    grouped_patches[prescription_id].append(img_path)
                except:
                    continue
            
            for pid, paths in grouped_patches.items():
                self.samples.append((paths, class_idx, pid))
                
        print(f"Found {len(self.samples)} prescriptions (bags) across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_paths, label, pid = self.samples[idx]
        images = []
        for p in patch_paths:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except Exception as e:
                print(f"Error loading {p}: {e}")
                continue
        
        if len(images) == 0:
            return torch.zeros((1, 3, 224, 224)), label, pid

        return torch.stack(images), label, pid

def collate_mil_pad(batch):
    batch_images, labels, pids = zip(*batch)
    max_pills = max([x.shape[0] for x in batch_images])
    c, h, w = batch_images[0].shape[1:]
    
    padded_batch = []
    mask = [] 
    
    for img_stack in batch_images:
        num_pills = img_stack.shape[0]
        pad_size = max_pills - num_pills
        
        m = torch.cat([torch.ones(num_pills), torch.zeros(pad_size)])
        mask.append(m)
        
        if pad_size > 0:
            padding = torch.zeros((pad_size, c, h, w))
            padded_batch.append(torch.cat([img_stack, padding], dim=0))
        else:
            padded_batch.append(img_stack)
            
    return torch.stack(padded_batch), torch.tensor(labels), torch.stack(mask), pids

# --- 3. Gated Attention MIL Model (FIXED FOR MIXED PRECISION) ---
class GatedAttentionMIL(nn.Module):
    def __init__(self, num_classes):
        super(GatedAttentionMIL, self).__init__()
        
        # Backbone (Will run in Float16 via Autocast)
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1]) 
        self.feature_dim = 2048
        
        # Heads (Will run in Float32)
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        self.bag_classifier = nn.Linear(self.feature_dim, num_classes)
        self.instance_classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, mask=None):
        B, M, C, H, W = x.shape
        
        # 1. Feature Extraction (Allowed in FP16)
        x = x.view(B * M, C, H, W) 
        features = self.feature_extractor(x) 
        features = features.view(B, M, -1)   
        
        # 2. Critical Math (Forced to Float32 to prevent Overflow)
        with torch.cuda.amp.autocast(enabled=False):
            # Cast features to float32 manually
            features = features.float()
            
            # --- Instance Prediction ---
            instance_logits = self.instance_classifier(features)
            
            # --- Attention Mechanism ---
            A_V = self.attention_V(features)
            A_U = self.attention_U(features)
            A = self.attention_weights(A_V * A_U) 
            
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                A = A.masked_fill(mask_expanded == 0, -1e9)
            
            # Softmax is unstable in FP16, now safe in FP32
            A = torch.softmax(A, dim=1) 
            
            # Weighted Sum
            bag_embedding = torch.sum(features * A, dim=1) 
            
            # Bag Prediction
            bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A
