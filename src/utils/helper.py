SUPPORTED_MODELS: dict[str, str] = {
    # ViT models
    "vit_tiny": "vit_tiny_patch16_224",
    "vit_small": "vit_small_patch16_224",
    "vit_base": "vit_base_patch16_224",
    "vit_large": "vit_large_patch16_224",
    "vit_huge": "vit_huge_patch14_224",
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
    """Display supported models and usage examples."""
    print("\n" + "=" * 60)
    print("SUPPORTED MODELS")
    print("=" * 60)

    print("\nVision Transformer (ViT):")
    print("  - vit_tiny")
    print("  - vit_small")
    print("  - vit_base")
    print("  - vit_large")
    print("  - vit_huge")

    print("\nSwin Transformer:")
    print("  - swin_tiny")
    print("  - swin_small")
    print("  - swin_base")
    print("  - swin_large")

    print("\nBEiT:")
    print("  - beit_base")
    print("  - beit_large")

    print("\n" + "=" * 60)
    print("Usage: python runner.py --model <model_name> [options]")
    print("Example: python runner.py --model vit_base --faultfree --metrics")
    print("=" * 60 + "\n")
