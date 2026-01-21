import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.ensemble import IsolationForest
import numpy as np
import sys

# --- CONFIG ---
BACKBONE_BATCH_SIZE = 256
# Contamination: The expected % of foreign objects. 
# 0.05 means we expect roughly 5% of pills could be wrong. 
# Adjusting this changes sensitivity.
CONTAMINATION = 0.05 

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {os.path.basename(weights_path)}...")
    model = models.resnet34(weights=None)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if 'fc.weight' in new_state_dict:
        model.fc = nn.Linear(512, new_state_dict['fc.weight'].shape[0])
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except:
        sys.exit("[ERROR] Backbone mismatch. Ensure you are using the correct .pth file.")

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def detect_anomalies(img_paths, backbone, device, transform):
    if not img_paths: return None
    
    # 1. Extract Features [N, 512]
    features_list = []
    
    # Batch processing to handle large bags
    for i in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
        batch_paths = img_paths[i : i+BACKBONE_BATCH_SIZE]
        images = []
        valid_indices_in_batch = []
        
        for idx, p in enumerate(batch_paths):
            try:
                with Image.open(p) as img:
                    images.append(transform(img.convert('RGB')))
                    valid_indices_in_batch.append(idx)
            except: pass
        
        if not images: continue
        
        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats = backbone(img_tensor)
            features_list.append(feats.float().cpu().numpy())
            
    if not features_list: return None
    
    # Combine features: [Total_Pills, 512]
    X = np.concatenate(features_list, axis=0)
    
    # 2. Run Isolation Forest
    # This finds the items that require the fewest "cuts" to isolate -> Outliers
    clf = IsolationForest(
        n_estimators=100, 
        contamination=CONTAMINATION, 
        random_state=42, 
        n_jobs=-1
    )
    
    # Predict: 1 = Normal, -1 = Anomaly
    preds = clf.fit_predict(X)
    
    # Get anomaly scores (lower is more anomalous)
    scores = clf.decision_function(X)
    
    foreign_objects = []
    
    for i in range(len(preds)):
        if preds[i] == -1: # It is an outlier
            foreign_objects.append({
                "file": os.path.basename(img_paths[i]),
                "score": f"{scores[i]:.4f}" # Negative scores are more anomalous
            })
            
    return foreign_objects

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Folder of images or Root folder")
    parser.add_argument("--backbone_path", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ONLY Backbone (No MIL head needed)
    backbone = load_backbone(args.backbone_path, device)
    
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Detect Mode
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    if direct_jpgs:
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        subdirs = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))
            
    print("\n" + "="*80)
    print(f"{'PRESCRIPTION ID':<30} | {'ANOMALIES DETECTED':<20}")
    print("-" * 80)
    
    for bag_id, img_paths in tasks:
        # Minimum pills check (Isolation Forest needs a population to work)
        if len(img_paths) < 5:
            print(f"{bag_id:<30} | [SKIP] Not enough pills ({len(img_paths)})")
            continue

        anomalies = detect_anomalies(img_paths, backbone, device, tfm)
        
        count = len(anomalies) if anomalies else 0
        status = f"YES ({count})" if count > 0 else "NO"
        
        print(f"{bag_id:<30} | {status:<20}")
        
        if anomalies:
            # Sort by most anomalous (lowest score)
            anomalies.sort(key=lambda x: float(x['score']))
            for fo in anomalies:
                print(f"   -> POTENTIAL FOREIGN OBJECT: {fo['file']} (Score: {fo['score']})")
            print("-" * 80)
            
    print("="*80)

if __name__ == "__main__":
    main()
