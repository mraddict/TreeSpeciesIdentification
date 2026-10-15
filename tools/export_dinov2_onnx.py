"""
export_dinov2_onnx.py
DINOv2 → ONNX export for DinoEmbedder.h (C++ / ONNX Runtime)

Usage:
    pip install torch torchvision onnx onnxruntime
    python export_dinov2_onnx.py --model dinov2_vits14 --out ../models/dinov2_vits14.onnx
    python export_dinov2_onnx.py --model dinov2_vitb14 --out ../models/dinov2_vitb14.onnx

Output graph:
    input  "image"     float32 [N, 3, 224, 224]   (ImageNet-normalized RGB)
    output "embedding" float32 [N, D]             (CLS token; D = 384 / 768 / 1024)

Model choice:
    vits14  (384-dim, ~22M params)  — 230k crops on CPU in roughly 1-2 h; good first pick
    vitb14  (768-dim, ~86M params)  — ~4x slower, modestly better separation
"""

import argparse
import numpy as np
import torch


class ClsOnly(torch.nn.Module):
    """Wraps DINOv2 so forward() returns only the CLS token [N, D]."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        out = self.backbone.forward_features(x)
        return out["x_norm_clstoken"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dinov2_vits14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    ap.add_argument("--out", default="dinov2_vits14.onnx")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    print(f"Loading {args.model} from torch.hub ...")
    backbone = torch.hub.load("facebookresearch/dinov2", args.model)
    model = ClsOnly(backbone).eval()

    dummy = torch.randn(1, 3, args.size, args.size)

    print(f"Exporting to {args.out} ...")
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,          # ← 추가: legacy exporter 사용
    )
    # Verify against PyTorch
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping verification.")
        return

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, args.size, args.size)

    with torch.no_grad():
        ref = model(x).numpy()

    got = sess.run(["embedding"], {"image": x.numpy()})[0]

    print(f"  output shape : {got.shape}")
    print(f"  max |diff|   : {np.abs(ref - got).max():.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
