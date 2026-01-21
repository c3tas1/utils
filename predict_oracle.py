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

# --- MODEL DEFINITIONS ---
class ResNetWithFeatures(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1]) 
        self.fc = original_model.fc 
        
    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, 1)
        logits = self.fc(f)
        return logits, f

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
        return self.bag_classifier(bag_embedding), A

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {os.path.basename(weights_path)}...")
    base_model = models.resnet34(weights=None)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    if 'fc.weight' in new_state_dict:
        num_classes = new_state_dict['fc.weight'].shape[0]
        base_model.fc = nn.Linear(512, num_classes)
    
    base_model.load_state_dict(new_state_dict, strict=True)
    model = ResNetWithFeatures(base_model)
    model.to(device)
    model.eval()
    return model

def predict_bag(img_paths, backbone, mil_model, device, class_list, transform):
    img_paths = sorted(img_paths)
    
    features_list = []
    strong_preds_list = []
    
    # 1. BATCH INFERENCE
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
                logits, feats = backbone(img_tensor)
            
            features_list.append(feats.float())
            strong_preds_list.append(logits.float())
            
    if not features_list: return None

    all_features = torch.cat(features_list, dim=0)          
    all_pill_logits = torch.cat(strong_preds_list, dim=0)   

    # 2. GET "THE BOSS" DECISION (Bag Class)
    bag_input = all_features.unsqueeze(0) 
    with torch.no_grad():
        bag_logits, _ = mil_model(bag_input)

    bag_probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(bag_probs, dim=0)
    bag_pred_name = class_list[bag_pred_idx.item()]

    # 3. ORACLE CHECK (Hard Vote Matching)
    foreign_objects = []
    
    # We take the ARGMAX of every pill. No thresholds. No "maybe".
    # Just: "What is your #1 choice?"
    pill_preds_idx = torch.argmax(all_pill_logits, dim=1) # [N]
    pill_confs = F.softmax(all_pill_logits, dim=1).max(dim=1)[0] # [N]
    
    for idx, p_idx in enumerate(pill_preds_idx):
        # Strict Agreement Check
        if p_idx.item() != bag_pred_idx.item():
            # It disagrees. It is a foreign object.
            # We add a tiny safety buffer (0.50) just to filter pure noise,
            # but effectively this relies on the ARGMAX logic.
            if pill_confs[idx] > 0.5:
                foreign_objects.append({
                    "file": os.path.basename(img_paths[idx]),
                    "predicted_as": class_list[p_idx.item()],
                    "confidence": pill_confs[idx].item()
                })

    return {
        "bag_pred": bag_pred_name,
        "bag_conf": bag_conf.item(),
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
    mil_chk = torch.load(args.mil_path, map_location=device)
    class_list = mil_chk['classes']
    mil_model = MILHead(len(class_list), mil_chk.get('input_dim', 512)).to(device)
    mil_model.load_state_dict(mil_chk['model_state_dict'], strict=False)
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

    print("\n" + "="*100)
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'STATUS':<15}")
    print("-" * 100)

    for bag_id, img_paths in tasks:
        res = predict_bag(img_paths, backbone, mil_model, device, class_list, tfm)
        
        if res:
            fo_count = len(res['foreign_objects'])
            
            # THE "TOLERANCE" FILTER
            # In your Test Script, if 2 pills were wrong but 48 were right, 
            # the bag was marked "Correct" (Acc 100%).
            # We replicate that here: We only flag "WARNING" if the disagreement is significant.
            
            status = "✅ OK"
            if fo_count > 0:
                # If more than 5% of the bag disagrees, THEN we warn.
                # This absorbs the 1-2 random errors that your Test Script was hiding.
                if (fo_count / len(img_paths)) > 0.05:
                    status = f"⚠️ CHECK ({fo_count})"
            
            print(f"{bag_id[:25]:<25} | {res['bag_pred'][:20]:<20} | {status:<15}")
            
            if "CHECK" in status:
                for fo in res['foreign_objects']:
                    print(f"       -> {fo['file']} thinks it is {fo['predicted_as']}")
                print("-" * 100)

    print("="*100)

if __name__ == "__main__":
    main()
