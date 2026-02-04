import torch
import torch.nn as nn
from torchvision import models
import argparse
import os
import sys
import subprocess

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
        # x shape: [Batch, Bag, 3, 224, 224]
        # For export, we assume Batch=1 to simplify dynamic axes logic
        # So input is effectively [1, N, 3, 224, 224]
        
        batch_size, bag_size, C, H, W = x.shape
        x = x.view(batch_size * bag_size, C, H, W)
        
        # Extract Features
        features = self.feature_extractor(x)
        features = features.view(batch_size, bag_size, -1) # [1, N, 512]
        
        # A. Instance Predictions
        flat_features = features.view(-1, self.input_dim)
        instance_logits = self.instance_classifier(flat_features) 
        instance_logits = instance_logits.view(batch_size, bag_size, -1)
        
        # B. Bag Predictions
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        A = self.attention_weights(A_V * A_U)
        A = torch.softmax(A, dim=1)
        
        bag_embedding = torch.sum(features * A, dim=1)
        bag_logits = self.bag_classifier(bag_embedding)
        
        # Return Bag Logits, Instance Logits, Attention Scores
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

    # Get class count
    class_list = checkpoint.get('classes')
    if not class_list:
        # Fallback: try to guess from weight shape if 'classes' key missing
        print("[WARN] 'classes' key missing in checkpoint. Guessing from weights...")
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        # Find bag_classifier.weight
        key = 'bag_classifier.weight' if 'bag_classifier.weight' in state_dict else 'module.bag_classifier.weight'
        num_classes = state_dict[key].shape[0]
        print(f"       Guessed {num_classes} classes.")
    else:
        num_classes = len(class_list)

    model = EndToEndMIL(num_classes)
    
    # Clean keys (Remove 'module.')
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_dict, strict=True)
    model.eval()

    # 2. Export to ONNX
    onnx_path = os.path.join(args.output_dir, "mil_model.onnx")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("--> Exporting to ONNX...")
    
    # Dummy Input: 1 Bag, 5 Pills, 3 Channels, 224x224
    dummy_input = torch.randn(1, 5, 3, 224, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        input_names=['input_bag'],
        output_names=['bag_logits', 'instance_logits', 'attention'],
        dynamic_axes={
            'input_bag': {1: 'num_pills'},         # The pill dimension (N) is dynamic
            'instance_logits': {1: 'num_pills'},   # Output scales with N
            'attention': {1: 'num_pills'}          # Attention scales with N
        }
    )
    print(f"--> ONNX saved to {onnx_path}")

    # 3. Convert ONNX to OpenVINO IR
    print("--> Converting to OpenVINO IR (FP16)...")
    
    # Construct the MO command
    # FP16 is recommended for inference speed on most Intel hardware
    cmd = [
        "mo",
        "--input_model", onnx_path,
        "--output_dir", args.output_dir,
        "--model_name", "mil_model_ir",
        "--data_type", "FP16" 
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"\n[SUCCESS] OpenVINO model saved to {args.output_dir}")
        print(f"FILES: mil_model_ir.xml, mil_model_ir.bin")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Model Optimizer failed. Ensure 'openvino-dev' is installed.")
        print("You can try running manually:")
        print(f"mo --input_model {onnx_path} --output_dir {args.output_dir} --data_type FP16")

if __name__ == "__main__":
    main()
