import torch
import torch.nn as nn
from torchvision import models
import argparse
import os
import sys
import openvino as ov  # New Native Import

# --- 1. DEFINE MODEL ARCHITECTURE (Must match training exactly) ---
class EndToEndMIL(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone
        base_model = models.resnet34(weights=None)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        
        # Heads
        self.input_dim = 512
        self.attention_V = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.input_dim, 128), nn.Sigmoid())
        self.attention_weights = nn.Linear(128, 1)
        
        self.bag_classifier = nn.Linear(self.input_dim, num_classes)
        self.instance_classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        # Input: [1, N, 3, 224, 224]
        batch_size, bag_size, C, H, W = x.shape
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract Features
        features = self.feature_extractor(x)
        features = features.view(batch_size, bag_size, -1) 
        
        # Instance Preds
        flat_features = features.view(-1, self.input_dim)
        instance_logits = self.instance_classifier(flat_features) 
        instance_logits = instance_logits.view(batch_size, bag_size, -1)
        
        # Bag Preds
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        bag_embedding = torch.sum(features * A, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        
        return bag_logits, instance_logits, A

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to mil_end_to_end_final_ddp.pth")
    parser.add_argument("--output_dir", type=str, default="./openvino_model", help="Directory to save IR files")
    args = parser.parse_args()

    # 1. Load PyTorch Model
    print(f"--> Loading model from {args.model_path}...")
    try:
        checkpoint = torch.load(args.model_path, map_location='cpu')
    except FileNotFoundError:
        sys.exit("Model file not found.")

    class_list = checkpoint.get('classes')
    if not class_list:
        print("[WARN] 'classes' key missing. Trying to infer...")
        state_dict = checkpoint['model_state_dict']
        key = 'bag_classifier.weight' if 'bag_classifier.weight' in state_dict else 'module.bag_classifier.weight'
        num_classes = state_dict[key].shape[0]
    else:
        num_classes = len(class_list)

    model = EndToEndMIL(num_classes)
    
    # Clean keys
    state_dict = checkpoint['model_state_dict']
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=True)
    model.eval()

    # 2. Export to ONNX
    # We still use ONNX as the bridge because it handles dynamic axes explicitly well.
    os.makedirs(args.output_dir, exist_ok=True)
    onnx_path = os.path.join(args.output_dir, "mil_model.onnx")
    
    print("--> Exporting to ONNX (with dynamic shapes)...")
    dummy_input = torch.randn(1, 5, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14, # Newer opset for 2024.x
        input_names=['input_bag'],
        output_names=['bag_logits', 'instance_logits', 'attention'],
        dynamic_axes={
            'input_bag': {1: 'num_pills'},         
            'instance_logits': {1: 'num_pills'},   
            'attention': {1: 'num_pills'}          
        }
    )
    print(f"--> ONNX saved to {onnx_path}")

    # 3. Convert to OpenVINO (The Modern Way)
    print("--> Converting to OpenVINO IR...")
    
    # Core OpenVINO conversion
    ov_model = ov.convert_model(onnx_path)
    
    # Save with FP16 Compression (This replaces --data_type FP16)
    ir_path = os.path.join(args.output_dir, "mil_model_ir.xml")
    
    # compress_to_fp16=True is the new way to get FP16 weights
    ov.save_model(ov_model, ir_path, compress_to_fp16=True)
    
    print(f"\n[SUCCESS] OpenVINO model saved to {ir_path}")
    print(f"Weights saved to {ir_path.replace('.xml', '.bin')}")
    print("Optimization: FP16 (Half Precision)")

if __name__ == "__main__":
    main()
