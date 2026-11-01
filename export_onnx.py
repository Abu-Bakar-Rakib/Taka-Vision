import torch
import torch.nn as nn
import timm

print("Loading checkpoint...")
ckpt = torch.load("best_banknote_model.pth", map_location="cpu", weights_only=False)
model_name = ckpt["model_name"]
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]

print(f"Model: {model_name}, classes: {class_names}, img_size: {img_size}")

base_model = timm.create_model(model_name, pretrained=False, num_classes=len(class_names))
base_model.load_state_dict(ckpt["model_state_dict"])
base_model.eval()

# Add Softmax layer so the ONNX output directly produces probabilities [0.0 - 1.0]
class WrappedModel(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        logits = self.m(x)
        return self.softmax(logits)

full_model = WrappedModel(base_model)
full_model.eval()

dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

print("Exporting to ONNX...")
torch.onnx.export(
    full_model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["probabilities"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "probabilities": {0: "batch_size"}
    },
    opset_version=14,
    dynamo=False
)

print("Testing exported ONNX model with onnx...")
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("SUCCESS: model.onnx is valid and ready!")

