import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import sys

# --- CONFIG ---
# Since the backbone is 98% accurate, we can lower the threshold slightly 
# because we trust it more.
FOREIGN_THRESH = 0.90   
BACKBONE_BATCH_SIZE = 256 

# --- 1. MIL MODEL (For Bag-Level Prediction) ---
class MILHead(nn.Module):
    def __init__(self, num_classes, input_dim=512):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        self.bag_classifier = nn.Linear(input_dim, num_classes)
        # We ignore the instance_classifier here because we will use the Backbone instead!
        self.instance_classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1) 
        bag_embedding = torch.sum(x * A, dim=1) 
        return self.bag_classifier(bag_embedding), A

# --- 2. BACKBONE WRAPPER (To get BOTH Features and Predictions) ---
class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1]) # All layers except FC
        self.fc = original_model.fc # The 98% accurate classifier
        
    def forward(self, x):
        # 1. Get the Features (512 dim)
        f = self.features(x)
        f = torch.flatten(f, 1)
        
        # 2. Get the Strong Prediction (Class dim)
        logits = self.fc(f)
        
        return logits, f

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {os.path.basename(weights_path)}...")
    base_model = models.resnet34(weights=None)
    
    # 1. Load Weights
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
    except FileNotFoundError:
        print(f"[ERROR] Backbone file not found: {weights_path}")
        sys.exit(1)

    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 2. Resize Head to match training classes
    if 'fc.weight' in new_state_dict:
        num_classes = new_state_dict['fc.weight'].shape[0]
        base_model.fc = nn.Linear(512, num_classes)
    
    # 3. Load State Dict
    try:
        base_model.load_state_dict(new_state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] Backbone Weights Mismatch:\n{e}")
        sys.exit(1)

    # 4. Wrap it to return both outputs
    model = ResNetWithFeatures(base_model)
    model.to(device)
    model.eval()
    return model

def predict_bag(img_paths, backbone, mil_model, device, class_list, transform):
    # Sort inputs
    img_paths = sorted(img_paths)
    
    # --- STEP 1: Run Backbone (Get Features AND Strong Predictions) ---
    features_list = []
    strong_preds_list = []
    
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
                # BACKBONE RETURNS TUPLE: (Logits, Features)
                logits, feats = backbone(img_tensor)
                
            features_list.append(feats.float())
            strong_preds_list.append(logits.float())
            
    if not features_list: return None

    # Aggregate
    all_features = torch.cat(features_list, dim=0)          # [N, 512]
    all_pill_logits = torch.cat(strong_preds_list, dim=0)   # [N, Classes]

    # --- STEP 2: Run MIL (Get Bag Prediction) ---
    bag_input = all_features.unsqueeze(0) # [1, N, 512]
    
    with torch.no_grad():
        bag_logits, _ = mil_model(bag_input)

    # Bag Result
    bag_probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(bag_probs, dim=0)
    bag_pred_name = class_list[bag_pred_idx.item()]

    # --- STEP 3: Foreign Object Check (Using 98% Backbone) ---
    foreign_objects = []
    
    # Convert Backbone Logits to Probs
    pill_probs = F.softmax(all_pill_logits, dim=1) # [N, Classes]
    p_confs, p_preds = torch.max(pill_probs, dim=1)
    
    for idx, (p_conf, p_pred_idx) in enumerate(zip(p_confs, p_preds)):
        
        # LOGIC: 
        # 1. Does the Pill (98% Acc) disagree with the Bag?
        if p_pred_idx.item() != bag_pred_idx.item():
            
            # 2. Is the Pill confident?
            if p_conf.item() > FOREIGN_THRESH:
                foreign_objects.append({
                    "file": os.path.basename(img_paths[idx]),
                    "predicted_as": class_list[p_pred_idx.item()],
                    "confidence": p_conf.item()
                })

    return {
        "bag_pred": bag_pred_name,
        "bag_conf": bag_conf.item(),
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Folder of images OR Folder of folders")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to resnet34_ddp_restored.pth")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to mil_final_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load MIL Head
    try:
        mil_chk = torch.load(args.mil_path, map_location=device)
    except FileNotFoundError:
        print(f"[ERROR] MIL Model not found: {args.mil_path}")
        sys.exit(1)
        
    class_list = mil_chk['classes']
    mil_model = MILHead(len(class_list), mil_chk.get('input_dim', 512)).to(device)
    # Strict=False to ignore the instance_classifier keys if they mismatch
    mil_model.load_state_dict(mil_chk['model_state_dict'], strict=False)
    mil_model.eval()

    # 2. Load Backbone (Dual-Head Mode)
    backbone = load_backbone(args.backbone_path, device)

    # 3. Prepare Data
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 4. Folder Scanning
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    tasks = []
    
    if len(direct_jpgs) > 0:
        print(f"--> Mode: Single Bag ({len(direct_jpgs)} images)")
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        print(f"--> Mode: Batch Processing")
        subdirs = sorted([os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))

    # 5. Prediction Loop
    print("\n" + "="*100)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'CONF':<8} | {'STATUS':<15}")
    print("-" * 100)

    for bag_id, img_paths in tasks:
        res = predict_bag(img_paths, backbone, mil_model, device, class_list, tfm)
        
        if res:
            fo_count = len(res['foreign_objects'])
            
            # Simple Status Display
            status = "✅ OK"
            if fo_count > 0:
                 status = f"⚠️ CHECK ({fo_count})"

            print(f"{bag_id[:25]:<25} | {res['bag_pred'][:20]:<20} | {res['bag_conf']:.1%}   | {status:<15}")
            
            # Detailed reporting only if flagged
            if fo_count > 0:
                print(f"   [!] Detailed Analysis for {bag_id}:")
                for fo in res['foreign_objects']:
                    print(f"       -> {fo['file']} looks like {fo['predicted_as']} ({fo['confidence']:.1%})")
                print("-" * 100)

    print("="*100)

if __name__ == "__main__":
    main()
