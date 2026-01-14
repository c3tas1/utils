import os
import glob
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F

# --- 1. Custom Dataset Handling Groups ---
class PrescriptionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects structure: root_dir/{className}/{className}_{imageName}_patch_{patch_no}.jpg
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] # List of (list_of_patch_paths, class_idx, prescription_id)
        
        # Get all class names
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Grouping logic
        print(f"Indexing dataset from {root_dir}...")
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            
            # Dictionary to group by unique prescription/image ID
            # Assumes format: {className}_{imageName}_patch_{patch_no}.jpg
            grouped_patches = {} 
            
            for img_path in glob.glob(os.path.join(cls_folder, "*.jpg")):
                filename = os.path.basename(img_path)
                try:
                    # Parse filename to get unique prescription ID (className + imageName)
                    parts = filename.split('_patch_')
                    prescription_id = parts[0] 
                    
                    if prescription_id not in grouped_patches:
                        grouped_patches[prescription_id] = []
                    grouped_patches[prescription_id].append(img_path)
                except:
                    continue
            
            # Add bags to samples
            for pid, paths in grouped_patches.items():
                self.samples.append((paths, class_idx, pid))
                
        print(f"Found {len(self.samples)} prescriptions (bags) across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch_paths, label, pid = self.samples[idx]
        
        images = []
        for p in patch_paths:
            img = Image.open(p).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Stack images: (Num_Pills, C, H, W)
        return torch.stack(images), label, pid

def collate_mil_pad(batch):
    """
    Collate function to handle variable number of pills per prescription.
    Pads to the max number of pills in the current batch.
    """
    batch_images, labels, pids = zip(*batch)
    
    # Find max pills in this batch
    max_pills = max([x.shape[0] for x in batch_images])
    c, h, w = batch_images[0].shape[1:]
    
    padded_batch = []
    mask = [] # 1 for real pill, 0 for padding
    
    for img_stack in batch_images:
        num_pills = img_stack.shape[0]
        pad_size = max_pills - num_pills
        
        # Create mask
        m = torch.cat([torch.ones(num_pills), torch.zeros(pad_size)])
        mask.append(m)
        
        if pad_size > 0:
            # Pad with zeros
            padding = torch.zeros((pad_size, c, h, w))
            padded_batch.append(torch.cat([img_stack, padding], dim=0))
        else:
            padded_batch.append(img_stack)
            
    return torch.stack(padded_batch), torch.tensor(labels), torch.stack(mask), pids


# --- 2. Gated Attention MIL Model ---
class GatedAttentionMIL(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(GatedAttentionMIL, self).__init__()
        
        # 1. Feature Extractor (Backbone)
        # Using ResNet50 as base, removing the final FC layer
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1]) # Output: 2048 dims
        self.feature_dim = 2048
        
        # 2. Gated Attention Mechanism
        # Determines which pills are "important" or "correct"
        self.attention_V = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.feature_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        # 3. Classifier Heads
        self.bag_classifier = nn.Linear(self.feature_dim, num_classes)
        
        # Auxiliary Head for individual pill accuracy monitoring
        self.instance_classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, mask=None):
        # x shape: (Batch, Max_Pills, C, H, W)
        B, M, C, H, W = x.shape
        
        # Collapse batch and pill dims for feature extraction
        x = x.view(B * M, C, H, W) 
        features = self.feature_extractor(x) # (B*M, 2048, 1, 1)
        features = features.view(B, M, -1)   # (B, M, 2048)
        
        # --- Instance Level Predictions (For Monitoring/Aux Loss) ---
        # We classify every single pill to satisfy your request for pill-level acc
        instance_logits = self.instance_classifier(features) # (B, M, Num_Classes)
        
        # --- Attention Pooling ---
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U) # (B, M, 1)
        
        # Handle Masking (ignore padding)
        if mask is not None:
            # Set attention of padding to -infinity so softmax makes them 0
            mask_expanded = mask.unsqueeze(-1) # (B, M, 1)
            A = A.masked_fill(mask_expanded == 0, -1e9)
            
        A = torch.softmax(A, dim=1) # Normalize weights across pills in the bag
        
        # Weighted Sum of Features
        bag_embedding = torch.sum(features * A, dim=1) # (B, 2048)
        
        # --- Bag Level Prediction ---
        bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A
