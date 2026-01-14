import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.checkpoint import checkpoint

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class PrescriptionDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_ram=False):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        self.cache_ram = cache_ram
        self.cache = {} # Simple RAM Cache
        
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
                    pid = parts[0] 
                    if pid not in grouped_patches: grouped_patches[pid] = []
                    grouped_patches[pid].append(img_path)
                except: continue
            
            for pid, paths in grouped_patches.items():
                self.samples.append((paths, class_idx, pid))
                
        print(f"Found {len(self.samples)} prescriptions.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_paths, label, pid = self.samples[idx]
        images = []
        
        # RAM Caching optimization for small datasets on big DGX nodes
        if self.cache_ram and idx in self.cache:
            return self.cache[idx], label, pid

        for p in patch_paths:
            try:
                img = Image.open(p).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except: continue
        
        if len(images) == 0:
            stack = torch.zeros((1, 3, 224, 224))
        else:
            stack = torch.stack(images)
            
        if self.cache_ram:
            self.cache[idx] = stack
            
        return stack, label, pid

def collate_mil_pad(batch):
    batch_images, labels, pids = zip(*batch)
    # Limit max pills to 64 during training to prevent OOM spikes
    # If a tray has 200 pills, we randomly sample 64 for training stability
    MAX_PILLS = 64 
    
    padded_batch = []
    mask = [] 
    
    for img_stack in batch_images:
        # Random Sampling if too big
        if img_stack.shape[0] > MAX_PILLS:
            perm = torch.randperm(img_stack.shape[0])[:MAX_PILLS]
            img_stack = img_stack[perm]
            
        num_pills = img_stack.shape[0]
        # We handle padding dynamically in the loop below
        padded_batch.append(img_stack)
        
    # Re-calculate max for this specific batch
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

class GatedAttentionMIL(nn.Module):
    def __init__(self, num_classes):
        super(GatedAttentionMIL, self).__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Split ResNet for checkpointing
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = 2048
        
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(self.feature_dim, num_classes)
        self.instance_classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, mask=None):
        B, M, C, H, W = x.shape
        x = x.view(B * M, C, H, W) 
        
        # --- GRADIENT CHECKPOINTING ---
        # This saves memory by re-computing activations during backward pass
        # Allows you to increase batch size significantly
        if self.training: 
            features = checkpoint(self.features, x, use_reentrant=False)
        else:
            features = self.features(x)
            
        features = features.view(B, M, -1)   
        
        with torch.cuda.amp.autocast(enabled=False):
            features = features.float()
            instance_logits = self.instance_classifier(features)
            A_V = self.attention_V(features)
            A_U = self.attention_U(features)
            A = self.attention_weights(A_V * A_U) 
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1)
                A = A.masked_fill(mask_expanded == 0, -1e9)
            A = torch.softmax(A, dim=1) 
            bag_embedding = torch.sum(features * A, dim=1) 
            bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A
