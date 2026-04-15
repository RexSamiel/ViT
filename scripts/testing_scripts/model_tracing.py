import torch
import timm
import torch.fx as fx
import operator

MODEL_NAME = "vit_base_patch16_224"
model = timm.create_model(MODEL_NAME, pretrained=False, img_size=224)

for m in model.modules():
    if hasattr(m, "fused_attn"):
        m.fused_attn = False

model.eval()

traced = fx.symbolic_trace(model)
print(f"\n{'#':<4} | {'OP TYPE':<15} | {'LABEL/MODULE'}")
print("-" * 65)

SKIPPED = {operator.getitem, getattr, "getattr"}
idx = 1

for node in traced.graph.nodes:
    if (
        node.op == "placeholder"
        or node.target in SKIPPED
        or "getitem" in str(node.target)
    ):
        continue

    target_str = str(node.target)
    label = target_str.split(".")[-1] if node.op != "call_module" else target_str

    suffix = ""
    if "softmax" in target_str.lower():
        suffix = " [SOFTMAX]"
    if "matmul" in target_str.lower() or node.target == operator.matmul:
        suffix = " [MATMUL]"
    if node.target == operator.add:
        suffix = " [RESIDUAL ADD]"

    print(f"{idx:<4} | {node.op:<15} | {label}{suffix}")
    idx += 1

dummy_input = torch.randn(1, 3, 224, 224)
onnx_filename = "vit_architecture.onnx"

print(f"\n--- Exporting to ONNX ---")

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=17,  # Changed from 14 to 17 to support modern LayerNorm
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    print(f"Success! File saved as: {onnx_filename}")
except Exception as e:
    print(f"Export failed: {e}")
    print("Trying fallback export...")
    torch.onnx.export(model, dummy_input, onnx_filename, opset_version=11)

print("\nTip: Drag 'vit_architecture.onnx' into https://netron.app")
