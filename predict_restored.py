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
    """
    Calculates class based on weighted average of individual pill predictions.
    Robust against abstract hallucinations.
    """
    # inst_logits: [1, Num_Pills, Num_Classes]
    # attention:   [1, Num_Pills, 1]
    
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0) 
    attn_weights = attention.squeeze(0)
    
    # Scale votes by attention (noise pills get near-zero vote)
    weighted_votes = inst_probs * attn_weights
    
    # Sum votes
    final_bag_score = torch.sum(weighted_votes, dim=0)
    
    conf, pred_idx = torch.max(final_bag_score, dim=0)
    return class_list[pred_idx.item()], conf.item()

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {weights_path}...")
    model = models.resnet34(weights=None)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
    except FileNotFoundError:
        print(f"[ERROR] Backbone file not found: {weights_path}")
        sys.exit(1)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean keys
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Auto-Resize Head to match weights (Critical for loading)
    if 'fc.weight' in new_state_dict:
        num_classes_in_ckpt = new_state_dict['fc.weight'].shape[0]
        model.fc = nn.Linear(512, num_classes_in_ckpt)
    
    # Strict Load
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Backbone Weights Mismatch:\n{e}")
        sys.exit(1)

    # Remove head for inference
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def predict_one_bag(img_paths, backbone, mil_model, device, class_list, transform):
    if not img_paths: return None
    
    # FIX 1: Sort paths to ensure deterministic behavior (Fixes 'Randomness')
    img_paths = sorted(img_paths)

    # 1. Extract Features
    features_list = []
    # Process in chunks
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
            features_list.append(feats.float()) # Cast to float32
            
    if not features_list: return None

    bag_features = torch.cat(features_list, dim=0).unsqueeze(0)

    # 2. MIL Inference
    with torch.no_grad():
        bag_logits, inst_logits, attn = mil_model(bag_features)
        
    # 3. Method A: Bag Classifier (Abstract)
    probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(probs, dim=0)
    bag_class = class_list[bag_pred_idx.item()]

    # 4. Method B: Weighted Vote (Robust)
    vote_class, vote_conf = get_weighted_vote(inst_logits, attn, class_list)

    # 5. Foreign Object Check (Using Vote Class as Ground Truth usually safer)
    # But standard MIL uses Bag Class. We will use Bag Class for consistency.
    foreign_objects = []
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0)
    p_confs, p_preds = torch.max(inst_probs, dim=1)
    
    for idx, (p_conf, p_pred) in enumerate(zip(p_confs, p_preds)):
        # FIX 2: Stricter Logic. 
        # Only flag if pill prediction is DIFFERENT from bag AND confidence > threshold
        if p_pred.item() != bag_pred_idx.item() and p_conf.item() > FOREIGN_THRESH:
            foreign_objects.append({
                "file": os.path.basename(img_paths[idx]),
                "predicted_as": class_list[p_pred.item()],
                "confidence": f"{p_conf.item():.4f}"
            })

    return {
        "bag_class": bag_class,
        "bag_conf": bag_conf.item(),
        "vote_class": vote_class,
        "vote_conf": vote_conf,
        "match": bag_class == vote_class,
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Folder of images OR Folder of folders")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to resnet34_ddp_restored.pth")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to mil_final_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Models
    print("--> Loading Models...")
    try:
        mil_chk = torch.load(args.mil_path, map_location=device)
    except FileNotFoundError:
        print(f"[ERROR] MIL Model file not found: {args.mil_path}")
        sys.exit(1)
        
    class_list = mil_chk['classes']
    mil_model = MILHead(len(class_list), mil_chk.get('input_dim', 512)).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'])
    mil_model.eval()
    
    backbone = load_backbone(args.backbone_path, device)
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Detect Mode
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    if len(direct_jpgs) > 0:
        print(f"--> Mode: Single Prescription ({len(direct_jpgs)} images)")
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        print(f"--> Mode: Batch Processing")
        subdirs = sorted([os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))
        print(f"--> Found {len(tasks)} prescriptions.")

    # Run Inference
    print("\n" + "="*110)
    print(f"{'ID':<20} | {'BAG PREDICTION':<20} | {'VOTE PREDICTION':<20} | {'AGREE?':<6} | {'FOREIGN OBJ':<12}")
    print("-" * 110)
    
    for bag_id, img_paths in tasks:
        res = predict_one_bag(img_paths, backbone, mil_model, device, class_list, tfm)
        if res:
            match_icon = "✅" if res['match'] else "⚠️"
            fo_status = "YES (!)" if res['foreign_objects'] else "NO"
            
            # Format output line
            line = f"{bag_id[:20]:<20} | {res['bag_class'][:18]:<18} ({int(res['bag_conf']*100)}%) | {res['vote_class'][:18]:<18} | {match_icon:<6} | {fo_status:<12}"
            print(line)
            
            # Print Details if warnings exist
            if res['foreign_objects'] or not res['match']:
                if not res['match']:
                    print(f"   [WARN] Disagreement! Individual pills look like '{res['vote_class']}' but bag vector looks like '{res['bag_class']}'.")
                
                if res['foreign_objects']:
                    print(f"   [WARN] {len(res['foreign_objects'])} Foreign Objects Detected:")
                    for fo in res['foreign_objects']:
                        print(f"       - {fo['file']} -> {fo['predicted_as']} ({float(fo['confidence']):.1%})")
                print("-" * 110)

    print("="*110)

if __name__ == "__main__":
    main()
