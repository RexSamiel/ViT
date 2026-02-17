import torch
import timm

model = timm.create_model("vit_tiny_patch16_224", pretrained=False)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "vit_tiny.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamo=False,  # IMPORTANT: forces stable legacy exporter
)

print("Exported vit_tiny.onnx successfully")
