import os
import glob
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

# --- CONFIG ---
FOREIGN_THRESH = 0.95  # Confidence threshold to flag a foreign object
BACKBONE_BATCH_SIZE = 256 # Batch size for feature extraction (adjust based on VRAM)

class MILHead(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(input_dim, num_classes)
        self.instance_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: [1, Pills, 512] (Batch size 1 for inference usually)
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1) 
        bag_embedding = torch.sum(x * A, dim=1) 
        return self.bag_classifier(bag_embedding), self.instance_classifier(x), A

class TestPrescriptionDataset(Dataset):
    def __init__(self, root_dir, class_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bags = [] 
        
        # Sort classes by length (Longest first) to ensure correct matching
        # e.g. Match "Vitamin_C" before matching "Vitamin"
        self.known_classes = sorted(class_list, key=len, reverse=True)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        
        print(f"--> Scanning test directory: {root_dir}")
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        matched_count = 0
        for folder_name in tqdm(subdirs, desc="Indexing"):
            found_class = None
            
            # Smart Matching Logic
            for cls in self.known_classes:
                if folder_name.startswith(cls):
                    # Verify boundary (ensure next char is _ or end of string)
                    remaining = folder_name[len(cls):]
                    if not remaining or remaining.startswith('_') or remaining.startswith(' '):
                        found_class = cls
                        break
            
            if found_class:
                rx_path = os.path.join(root_dir, folder_name)
                img_paths = glob.glob(os.path.join(rx_path, "*.jpg"))
                
                if len(img_paths) > 0:
                    self.bags.append({
                        "rx_id": folder_name,
                        "gt_class": found_class,
                        "gt_idx": self.class_to_idx[found_class],
                        "paths": img_paths
                    })
                    matched_count += 1
        
        print(f"--> Indexed {matched_count} prescriptions.")

    def __len__(self): return len(self.bags)

    def __getitem__(self, idx):
        # Returns metadata; images are loaded in the main loop to handle variable bag sizes
        return self.bags[idx]

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone: {os.path.basename(weights_path)}")
    model = models.resnet34(weights=None)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Auto-adjust head to load weights, then remove it
    if 'fc.weight' in new_state_dict:
        num_classes = new_state_dict['fc.weight'].shape[0]
        model.fc = nn.Linear(512, num_classes)
        
    model.load_state_dict(new_state_dict, strict=False)
    model.fc = nn.Identity() # Remove head to get features
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test folder containing prescription folders")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to resnet34_ddp_restored.pth")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to mil_final_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load MIL Head & Class List
    print(f"--> Loading MIL Head: {os.path.basename(args.mil_path)}")
    mil_chk = torch.load(args.mil_path, map_location=device)
    class_list = mil_chk['classes']
    input_dim = mil_chk.get('input_dim', 512)
    
    mil_model = MILHead(num_classes=len(class_list), input_dim=input_dim).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'])
    mil_model.eval()
    
    # 2. Load Backbone
    backbone = load_backbone(args.backbone_path, device)

    # 3. Setup Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    ds = TestPrescriptionDataset(args.test_dir, class_list)
    
    # 4. Inference Loop
    correct = 0
    total = 0
    foreign_objects = []
    
    print("\n--> Starting Inference...")
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            bag_data = ds[i]
            img_paths = bag_data['paths']
            gt_idx = bag_data['gt_idx']
            rx_id = bag_data['rx_id']
            
            # A. Extract Features (Batched to prevent OOM)
            features_list = []
            
            # Create a temporary loader for just this bag's images
            # (Manual batching is simpler here than a full DataLoader for variable sizes)
            for chunk_start in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
                chunk_paths = img_paths[chunk_start : chunk_start + BACKBONE_BATCH_SIZE]
                images = []
                valid_chunk_paths = []
                
                for p in chunk_paths:
                    try:
                        with Image.open(p) as img:
                            images.append(tfm(img.convert('RGB')))
                            valid_chunk_paths.append(p)
                    except: pass
                
                if not images: continue
                
                img_tensor = torch.stack(images).to(device)
                
                # Backbone Inference
                with torch.cuda.amp.autocast():
                    feats = backbone(img_tensor)
                features_list.append(feats)
            
            if not features_list:
                print(f"[Warning] No valid images in {rx_id}")
                continue
                
            # Combine all features for the bag [1, Total_Pills, 512]
            bag_features = torch.cat(features_list, dim=0).unsqueeze(0) 
            
            # B. MIL Inference
            bag_logits, inst_logits, attention = mil_model(bag_features)
            
            # C. Bag Level Metrics
            pred_idx = torch.argmax(bag_logits, dim=1).item()
            if pred_idx == gt_idx:
                correct += 1
            total += 1
            
            # D. Foreign Object Detection
            # inst_logits shape: [1, Total_Pills, Num_Classes]
            probs = F.softmax(inst_logits, dim=2).squeeze(0) # [Total_Pills, Num_Classes]
            
            # Get max prob and class for each pill
            p_confs, p_preds = torch.max(probs, dim=1)
            
            for p_idx, (conf, pred) in enumerate(zip(p_confs, p_preds)):
                # Logic: If pill prediction != Bag Prediction AND Confidence is high
                if pred.item() != pred_idx and conf.item() > FOREIGN_THRESH:
                    foreign_objects.append({
                        "Prescription": rx_id,
                        "File": os.path.basename(img_paths[p_idx]),
                        "Predicted_Bag": class_list[pred_idx],
                        "Predicted_Pill": class_list[pred.item()],
                        "Confidence": f"{conf.item():.4f}"
                    })

    # 5. Final Report
    acc = correct / total if total > 0 else 0
    print("\n" + "="*50)
    print(f"FINAL TEST ACCURACY: {acc:.2%}")
    print(f"Total Prescriptions: {total}")
    print(f"Foreign Objects Detected: {len(foreign_objects)}")
    print("="*50)
    
    if foreign_objects:
        df = pd.DataFrame(foreign_objects)
        df.to_csv("test_foreign_objects.csv", index=False)
        print("--> Foreign object details saved to 'test_foreign_objects.csv'")

if __name__ == "__main__":
    main()
