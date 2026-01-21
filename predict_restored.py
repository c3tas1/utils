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
BACKBONE_BATCH_SIZE = 256
# CRITICAL FIX: Higher threshold prevents flagging valid but "uncertain" pills
# Only flag a pill as Foreign if the model is SUPER sure it belongs to a DIFFERENT class.
FOREIGN_CONF_THRESH = 0.95 

# --- MODEL DEFINITIONS (Original Voting/Attention) ---
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

def load_backbone(weights_path, device):
    # Strict loading logic to match your Test script
    print(f"--> Loading Backbone...")
    model = models.resnet34(weights=None)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Auto-match head
    if 'fc.weight' in new_state_dict:
        model.fc = nn.Linear(512, new_state_dict['fc.weight'].shape[0])
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        print(f"[ERROR] Backbone weights mismatch: {e}")
        sys.exit(1)

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def predict_bag(img_paths, backbone, mil_model, device, class_list, transform):
    if not img_paths: return None
    
    # Sort paths to ensure deterministic behavior (Fixes 'Randomness')
    img_paths = sorted(img_paths)

    # 1. Extract Features (Exactly as Test Script)
    features_list = []
    for i in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
        batch_paths = img_paths[i : i+BACKBONE_BATCH_SIZE]
        images = []
        valid_indices = []
        for idx, p in enumerate(batch_paths):
            try:
                with Image.open(p) as img:
                    images.append(transform(img.convert('RGB')))
                    valid_indices.append(idx)
            except: pass
        
        if not images: continue
        
        img_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                feats = backbone(img_tensor)
            features_list.append(feats.float()) # Ensure Float32
            
    if not features_list: return None

    # [1, N, 512]
    bag_features = torch.cat(features_list, dim=0).unsqueeze(0)

    # 2. Inference (Original MIL Logic)
    with torch.no_grad():
        bag_logits, inst_logits, attn = mil_model(bag_features)
        
    # 3. Bag Prediction (The "Gold Standard")
    bag_probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(bag_probs, dim=0)
    bag_class = class_list[bag_pred_idx.item()]

    # 4. Foreign Object Detection (Strict Logic)
    foreign_objects = []
    
    # Get predictions for every individual pill
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0) # [N, Classes]
    p_confs, p_preds = torch.max(inst_probs, dim=1)
    
    for i in range(len(p_confs)):
        p_class_idx = p_preds[i].item()
        p_conf = p_confs[i].item()
        
        # LOGIC FIX:
        # Only flag if:
        # A) The pill prediction is DIFFERENT from the bag
        # B) AND the pill classifier is VERY CONFIDENT (e.g. > 95%) that it is different
        # This ignores "low confidence confusion" which was causing your False Positives.
        
        if p_class_idx != bag_pred_idx.item():
            if p_conf > FOREIGN_CONF_THRESH:
                foreign_objects.append({
                    "file": os.path.basename(img_paths[i]),
                    "predicted_as": class_list[p_class_idx],
                    "confidence": p_conf
                })

    return {
        "class": bag_class,
        "conf": bag_conf.item(),
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--backbone_path", type=str, required=True)
    parser.add_argument("--mil_path", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    print("--> Loading Models...")
    mil_chk = torch.load(args.mil_path, map_location=device)
    class_list = mil_chk['classes']
    
    # Use standard MILHead (Not Transformer)
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
    if direct_jpgs:
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        subdirs = sorted([os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs: tasks.append((os.path.basename(d), imgs))

    print("\n" + "="*80)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'CONF':<8} | {'FOREIGN OBJ':<12}")
    print("-" * 80)
    
    for bag_id, img_paths in tasks:
        res = predict_bag(img_paths, backbone, mil_model, device, class_list, tfm)
        if res:
            fo_status = "YES (!)" if res['foreign_objects'] else "NO"
            print(f"{bag_id[:25]:<25} | {res['class'][:20]:<20} | {res['conf']:.2%}   | {fo_status:<12}")
            
            if res['foreign_objects']:
                print(f"   >>> Found {len(res['foreign_objects'])} Disagreeing Pills:")
                for fo in res['foreign_objects']:
                    print(f"       - {fo['file']} -> {fo['predicted_as']} ({fo['confidence']:.1%})")
                print("-" * 80)
    print("="*80)

if __name__ == "__main__":
    main()
