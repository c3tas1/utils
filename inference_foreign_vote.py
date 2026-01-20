import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import pandas as pd
import sys
from tqdm import tqdm

# --- CONFIG ---
FOREIGN_THRESH = 0.95   # Confidence to flag a pill as "Wrong"
BACKBONE_BATCH_SIZE = 256 

# --- MODEL DEFINITIONS ---
class MILHead(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(input_dim, num_classes)
        self.instance_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1) 
        bag_embedding = torch.sum(x * A, dim=1) 
        return self.bag_classifier(bag_embedding), self.instance_classifier(x), A

def get_weighted_vote(inst_logits, attention, class_list):
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0) 
    attn_weights = attention.squeeze(0)
    weighted_votes = inst_probs * attn_weights
    final_bag_score = torch.sum(weighted_votes, dim=0)
    conf, pred_idx = torch.max(final_bag_score, dim=0)
    return class_list[pred_idx.item()], conf.item()

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {os.path.basename(weights_path)}...")
    model = models.resnet34(weights=None)
    
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
    except FileNotFoundError:
        print(f"[ERROR] Backbone file not found: {weights_path}")
        sys.exit(1)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Auto-Resize Head to match weights
    if 'fc.weight' in new_state_dict:
        num_classes_in_ckpt = new_state_dict['fc.weight'].shape[0]
        model.fc = nn.Linear(512, num_classes_in_ckpt)
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Backbone Weights Mismatch:\n{e}")
        sys.exit(1)

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

# --- DATASET PARSING ---
class TestDataset:
    def __init__(self, root_dir, known_classes):
        self.bags = []
        self.known_classes = sorted(known_classes, key=len, reverse=True) # Longest match first
        
        print(f"--> Scanning test folders in {root_dir}...")
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for folder in subdirs:
            # Smart Parse: "Amoxicillin_Rx1" -> "Amoxicillin"
            gt_class = None
            for cls in self.known_classes:
                if folder.startswith(cls):
                    # Check boundary (e.g. ensure 'Vitamin C' doesn't match 'Vitamin B')
                    suffix = folder[len(cls):]
                    if not suffix or suffix[0] in ['_', ' ', '-', '.']:
                        gt_class = cls
                        break
            
            if gt_class:
                full_path = os.path.join(root_dir, folder)
                imgs = glob.glob(os.path.join(full_path, "*.jpg"))
                if imgs:
                    self.bags.append({
                        "id": folder,
                        "gt": gt_class,
                        "paths": imgs
                    })
            else:
                print(f"    [Warning] Could not map folder '{folder}' to a known class. Skipping.")

        print(f"--> Found {len(self.bags)} valid test prescriptions.")

    def __len__(self): return len(self.bags)
    def __getitem__(self, i): return self.bags[i]

def evaluate_bag(bag_data, backbone, mil_model, device, class_list, transform):
    img_paths = bag_data['paths']
    
    # 1. Feature Extraction
    features_list = []
    for i in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
        batch_paths = img_paths[i : i + BACKBONE_BATCH_SIZE]
        images = []
        for p in batch_paths:
            try:
                with Image.open(p) as img:
                    images.append(transform(img.convert('RGB')))
            except: pass
        
        if not images: continue
        img_tensor = torch.stack(images).to(device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats = backbone(img_tensor)
            features_list.append(feats.float()) # Cast to FP32
            
    if not features_list: return None

    bag_features = torch.cat(features_list, dim=0).unsqueeze(0)

    # 2. MIL Prediction
    with torch.no_grad():
        bag_logits, inst_logits, attn = mil_model(bag_features)

    # 3. Get Predictions
    # Method A: Bag Classifier
    probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(probs, dim=0)
    bag_pred = class_list[bag_pred_idx.item()]

    # Method B: Weighted Vote
    vote_pred, vote_conf = get_weighted_vote(inst_logits, attn, class_list)

    return {
        "gt": bag_data['gt'],
        "bag_pred": bag_pred,
        "vote_pred": vote_pred,
        "bag_conf": bag_conf.item(),
        "correct": bag_pred == bag_data['gt']
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test folder with Class_RxID structure")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to resnet34_ddp_restored.pth")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to mil_final_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load MIL Head & Classes
    try:
        mil_chk = torch.load(args.mil_path, map_location=device)
    except FileNotFoundError:
        print(f"[ERROR] MIL Model not found: {args.mil_path}")
        sys.exit(1)
        
    class_list = mil_chk['classes']
    mil_model = MILHead(len(class_list), mil_chk.get('input_dim', 512)).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'])
    mil_model.eval()

    # 2. Load Backbone
    backbone = load_backbone(args.backbone_path, device)

    # 3. Prepare Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TestDataset(args.test_dir, class_list)
    if len(dataset) == 0:
        print("No valid folders found. Exiting.")
        sys.exit(1)

    # 4. Evaluation Loop
    total = 0
    correct_bag = 0
    correct_vote = 0
    
    results = []

    print("\n" + "="*90)
    print(f"{'ID':<25} | {'GT CLASS':<20} | {'PREDICTION':<20} | {'STATUS':<8}")
    print("-" * 90)

    for i in tqdm(range(len(dataset)), desc="Testing"):
        bag_data = dataset[i]
        res = evaluate_bag(bag_data, backbone, mil_model, device, class_list, tfm)
        
        if res:
            total += 1
            is_correct = res['bag_pred'] == res['gt']
            if is_correct: correct_bag += 1
            if res['vote_pred'] == res['gt']: correct_vote += 1
            
            status = "✅" if is_correct else "❌"
            
            # Print only errors or every Nth line to keep log clean? 
            # Let's print everything for small test sets, or errors only for large.
            # Here we print everything neatly.
            print(f"{bag_data['id'][:25]:<25} | {res['gt'][:20]:<20} | {res['bag_pred'][:20]:<20} | {status:<8}")
            
            if not is_correct:
                results.append(f"FAIL: {bag_data['id']} (Pred: {res['bag_pred']}, Vote: {res['vote_pred']})")

    # 5. Final Report
    bag_acc = correct_bag / total if total > 0 else 0
    vote_acc = correct_vote / total if total > 0 else 0
    
    print("\n" + "="*50)
    print("TEST REPORT")
    print("="*50)
    print(f"Total Prescriptions: {total}")
    print(f"Bag-Level Accuracy:  {bag_acc:.2%}  ({correct_bag}/{total})")
    print(f"Voting Accuracy:     {vote_acc:.2%}  ({correct_vote}/{total})")
    print("="*50)
    
    if len(results) > 0:
        print("\nFailures:")
        for r in results:
            print(r)

if __name__ == "__main__":
    main()
