SUPPORTED_MODELS: dict[str, str] = {
    # ViT models
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
    # DeiT models
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_small_distilled": "deit_small_distilled_patch16_224",
    "deit_base": "deit_base_patch16_224",
    "deit_base_distilled": "deit_base_distilled_patch16_224",
    # Swin models
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_small": "swin_small_patch4_window7_224",
    "swin_base": "swin_base_patch4_window7_224",
    "swin_large": "swin_large_patch4_window7_224",
    # BEiT models
    "beit_base": "beit_base_patch16_224",
    "beit_large": "beit_large_patch16_224",
}


def print_supported_models() -> None:
    print(f"""
============================================================
SUPPORTED MODELS
============================================================

Vision Transformer (ViT):
  - vit_tiny
  - vit_small
  - vit_base
  - vit_large
  - vit_huge

DeiT (Data-efficient Image Transformers):
  - deit_tiny
  - deit_small
  - deit_small_distilled
  - deit_base
  - deit_base_distilled

Swin Transformer:
  - swin_tiny
  - swin_small
  - swin_base
  - swin_large

BEiT:
  - beit_base
  - beit_large

============================================================
Usage: python runner.py --model <model_name> [options]
Example: python runner.py --model vit_base --faultfree --metrics
============================================================
""")

