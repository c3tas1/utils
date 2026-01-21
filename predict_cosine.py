import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import sys

# --- CONFIG ---
BACKBONE_BATCH_SIZE = 256
# Pills with cosine similarity < 0.5 to the bag center are flagged
SIMILARITY_THRESH = 0.5 

# --- MODEL CLASS (Must match training) ---
class TransformerMIL(nn.Module):
    def __init__(self, num_classes, input_dim=512, hidden_dim=512, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, 
            dim_feedforward=hidden_dim*2, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask=None):
        h = self.relu(self.fc_in(x))
        key_padding_mask = (mask == 0) if mask is not None else None
        h_trans = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h_trans = self.norm(h_trans)
        attn_logits = self.attn_layer(h_trans)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        bag_embedding = torch.sum(h_trans * attn_weights, dim=1)
        logits = self.classifier(bag_embedding)
        return logits, h_trans, bag_embedding, attn_weights

def load_backbone(weights_path, device):
    print(f"--> Loading Backbone from {os.path.basename(weights_path)}...")
    model = models.resnet34(weights=None)
    try:
        checkpoint = torch.load(weights_path, map_location='cpu')
    except:
        sys.exit("[ERROR] Backbone file not found")
        
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if 'fc.weight' in new_state_dict:
        model.fc = nn.Linear(512, new_state_dict['fc.weight'].shape[0])
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
    except Exception as e:
        sys.exit(f"[ERROR] Backbone mismatch: {e}")

    model.fc = nn.Identity()
    model.to(device)
    model.eval()
    return model

def predict_bag(img_paths, backbone, trans_model, device, class_list, transform):
    if not img_paths: return None
    
    # 1. Backbone Extraction
    features_list = []
    for i in range(0, len(img_paths), BACKBONE_BATCH_SIZE):
        batch_paths = img_paths[i : i+BACKBONE_BATCH_SIZE]
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
            features_list.append(feats.float())
            
    if not features_list: return None
    bag_features = torch.cat(features_list, dim=0).unsqueeze(0) # [1, N, 512]
    
    # 2. Transformer Inference
    with torch.no_grad():
        bag_logits, h_pills, h_bag, attn = trans_model(bag_features)
        
    # 3. Bag Prediction
    probs = F.softmax(bag_logits, dim=1).squeeze()
    bag_conf, bag_pred_idx = torch.max(probs, dim=0)
    bag_class = class_list[bag_pred_idx.item()]
    
    # 4. Outlier Detection (Cosine Similarity)
    # h_pills: [1, N, 512] -> [N, 512]
    # h_bag:   [1, 512]    -> [1, 512]
    pills_vecs = h_pills.squeeze(0)
    bag_vec = h_bag.squeeze(0).unsqueeze(0) # [1, 512]
    
    # Normalize
    pills_norm = F.normalize(pills_vecs, p=2, dim=1)
    bag_norm = F.normalize(bag_vec, p=2, dim=1)
    
    # Dot Product (since normalized, this is Cosine Similarity)
    # Shape: [N, 1]
    similarities = torch.mm(pills_norm, bag_norm.t()).squeeze(1)
    
    foreign_objects = []
    
    # To properly map indices back to files, we must account for any images that failed to load.
    # Assuming all loaded fine for simplicity, but strictly we should track valid indices.
    # The `img_paths` list corresponds 1:1 to features if no load errors occurred.
    
    valid_idx = 0
    for i, path in enumerate(img_paths):
        # Safety check if we have enough features (in case of load errors)
        if valid_idx >= len(similarities): break
            
        score = similarities[valid_idx].item()
        valid_idx += 1
        
        if score < SIMILARITY_THRESH:
            foreign_objects.append({
                "file": os.path.basename(path),
                "score": score
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
    parser.add_argument("--mil_path", type=str, required=True, help="mil_transformer_model.pth")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    print("--> Loading Transformer Model...")
    chk = torch.load(args.mil_path, map_location=device)
    class_list = chk['classes']
    
    model = TransformerMIL(
        len(class_list), 
        hidden_dim=chk['hidden_dim'], 
        n_layers=chk['n_layers'], 
        n_heads=chk['n_heads']
    ).to(device)
    model.load_state_dict(chk['model_state_dict'])
    model.eval()
    
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
    print(f"{'ID':<25} | {'PREDICTION':<20} | {'CONF':<8} | {'FOREIGN OBJ':<12}")
    print("-" * 80)
    
    for bag_id, img_paths in tasks:
        res = predict_bag(img_paths, backbone, model, device, class_list, tfm)
        if res:
            fo_status = "YES (!)" if res['foreign_objects'] else "NO"
            print(f"{bag_id[:25]:<25} | {res['class'][:20]:<20} | {res['conf']:.2%}   | {fo_status:<12}")
            
            if res['foreign_objects']:
                print(f"   >>> Found {len(res['foreign_objects'])} Anomalies (Low Similarity):")
                # Sort by lowest score (most anomalous) first
                res['foreign_objects'].sort(key=lambda x: x['score'])
                for fo in res['foreign_objects'][:5]: # Show top 5
                    print(f"       - {fo['file']} (Sim: {fo['score']:.2f})")
                if len(res['foreign_objects']) > 5: print("       ... and others.")
                print("-" * 80)
                
    print("="*80)

if __name__ == "__main__":
    main()
