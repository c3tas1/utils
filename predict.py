import os
import glob
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import pandas as pd

# --- CONFIG ---
FOREIGN_THRESH = 0.95  # Confidence to flag a pill as "Wrong"
BACKBONE_BATCH_SIZE = 256 # Adjust based on GPU VRAM

# --- MODEL DEFINITIONS (Must match training) ---
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
    print(f"--> Loading Backbone...")
    model = models.resnet34(weights=None)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if 'fc.weight' in new_state_dict:
        num_classes = new_state_dict['fc.weight'].shape[0]
        model.fc = nn.Linear(512, num_classes)
        
    model.load_state_dict(new_state_dict, strict=False)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def predict_one_bag(img_paths, backbone, mil_model, device, class_list, transform):
    """
    Runs prediction on a list of image paths (representing one bag/prescription)
    """
    if not img_paths:
        return None

    # 1. Extract Features (Batched)
    features_list = []
    
    # Process in chunks to avoid OOM
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
            # CRITICAL: Cast back to float32
            features_list.append(feats.float())
            
    if not features_list:
        return None

    # [1, Total_Pills, 512]
    bag_features = torch.cat(features_list, dim=0).unsqueeze(0)

    # 2. MIL Prediction
    with torch.no_grad():
        bag_logits, inst_logits, attn = mil_model(bag_features)
        
    # 3. Process Results
    probs = F.softmax(bag_logits, dim=1).squeeze()
    conf, pred_idx = torch.max(probs, dim=0)
    predicted_class = class_list[pred_idx.item()]
    
    # 4. Foreign Object Check
    foreign_objects = []
    inst_probs = F.softmax(inst_logits, dim=2).squeeze(0) # [Pills, Classes]
    p_confs, p_preds = torch.max(inst_probs, dim=1)
    
    for idx, (p_conf, p_pred) in enumerate(zip(p_confs, p_preds)):
        if p_pred.item() != pred_idx.item() and p_conf.item() > FOREIGN_THRESH:
            foreign_objects.append({
                "file": os.path.basename(img_paths[idx]),
                "predicted_as": class_list[p_pred.item()],
                "confidence": f"{p_conf.item():.4f}"
            })

    return {
        "class": predicted_class,
        "confidence": conf.item(),
        "foreign_objects": foreign_objects
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to a single folder of images OR a root folder of bags")
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to resnet34_ddp_restored.pth")
    parser.add_argument("--mil_path", type=str, required=True, help="Path to mil_final_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    mil_chk = torch.load(args.mil_path, map_location=device)
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

    # 2. Detect Input Mode (Single Bag vs Batch)
    # Check if input_path contains jpgs directly
    direct_jpgs = glob.glob(os.path.join(args.input_path, "*.jpg"))
    
    tasks = []
    if len(direct_jpgs) > 0:
        # Mode A: Single Bag
        print(f"--> Detected Single Bag Mode ({len(direct_jpgs)} images)")
        tasks.append((os.path.basename(args.input_path), direct_jpgs))
    else:
        # Mode B: Root Folder containing subfolders
        print(f"--> Detected Batch Mode (Scanning subfolders in {args.input_path})")
        subdirs = [os.path.join(args.input_path, d) for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))]
        for d in subdirs:
            imgs = glob.glob(os.path.join(d, "*.jpg"))
            if imgs:
                tasks.append((os.path.basename(d), imgs))
        print(f"--> Found {len(tasks)} bags to process.")

    # 3. Run Inference
    results = []
    print("\n" + "="*60)
    print(f"{'PRESCRIPTION ID':<30} | {'PREDICTION':<20} | {'CONF %':<8}")
    print("-" * 60)
    
    for bag_id, img_paths in tasks:
        res = predict_one_bag(img_paths, backbone, mil_model, device, class_list, tfm)
        if res:
            print(f"{bag_id:<30} | {res['class']:<20} | {res['confidence']:.2%}")
            
            if res['foreign_objects']:
                print(f"   [!] WARNING: {len(res['foreign_objects'])} Foreign Objects detected in {bag_id}")
                for fo in res['foreign_objects']:
                    print(f"       -> {fo['file']} looks like {fo['predicted_as']} ({float(fo['confidence']):.0%})")
            
            results.append({"ID": bag_id, "Prediction": res['class'], "Confidence": res['confidence']})

    print("="*60)

if __name__ == "__main__":
    main()
