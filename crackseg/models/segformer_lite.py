from __future__ import annotations

"""SegFormer-Lite: lightweight transfer model with timm encoder.

Notes
- Uses timm backbones with `features_only=True` to obtain multi-scale features.
- Supports encoder aliases: "segformer_b0" -> "mit_b0", "mobilenet_v3_small" -> "mobilenetv3_small_100".
- Simple top-down decoder: 1x1 reduce -> upsample -> conv3x3 fuse, final 1-channel logits.
- If pretrained weights cannot be downloaded (offline), falls back to random init with a warning.
"""

from typing import List, Dict, Optional
import torch
import torch.nn as nn
import logging


LOGGER = logging.getLogger(__name__)


def _resolve_encoder_name(name: str) -> str:
    n = name.lower()
    aliases: Dict[str, str] = {
        "segformer_b0": "mit_b0",
        "mobilenet_v3_small": "mobilenetv3_small_100",
        "mobilenetv3_small": "mobilenetv3_small_100",
    }
    return aliases.get(n, name)


def _try_create_features_model(
    timm_mod,
    name: str,
    in_chans: int,
    pretrained: bool,
) -> Optional[nn.Module]:
    try:
        return timm_mod.create_model(name, features_only=True, pretrained=bool(pretrained), in_chans=in_chans)
    except Exception:
        try:
            # Retry without pretrained in case of download/offline issues or missing weights
            return timm_mod.create_model(name, features_only=True, pretrained=False, in_chans=in_chans)
        except Exception:
            return None


class Conv1x1(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Conv3x3(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SegFormerLite(nn.Module):
    def __init__(
        self,
        encoder_name: str = "segformer_b0",
        pretrained: bool = True,
        in_ch: int = 3,
        num_classes: int = 1,
        base: int = 32,
    ):
        super().__init__()
        try:
            import timm  # type: ignore
        except Exception as e:
            raise ImportError(
                "timm is required for segformer_lite. Install with `pip install timm`."
            ) from e

        # Build a list of candidate encoder names to try in order
        enc_name_req = _resolve_encoder_name(encoder_name)
        candidates: List[str] = [enc_name_req]
        # Also try the raw provided name, in case aliasing wasn't needed
        if encoder_name not in candidates:
            candidates.append(encoder_name)
        # If requesting a segformer variant but not available, try MobileNetV3 small/large as fallback
        if any(k in enc_name_req.lower() for k in ["mit_b", "segformer", "mix_transformer", "mit_"]):
            for alt in ("mobilenetv3_small_100", "mobilenetv3_large_100"):
                if alt not in candidates:
                    candidates.append(alt)

        # If user requested mobilenet aliases, ensure canonical name is in list
        if "mobilenet_v3_small" in encoder_name.lower() and "mobilenetv3_small_100" not in candidates:
            candidates.insert(0, "mobilenetv3_small_100")

        # Also try segformer_b0 literal in case this timm build uses that name
        if "segformer_b0" not in candidates:
            candidates.append("segformer_b0")

        chosen = None
        for cand in candidates:
            enc = _try_create_features_model(timm, cand, in_chans=in_ch, pretrained=bool(pretrained))
            if enc is not None:
                chosen = (cand, enc)
                break
        if chosen is None:
            # Last resort: enumerate timm models and pick a generic lightweight encoder
            try:
                avail = timm.list_models(pretrained=False)
            except Exception:
                avail = []
            fallback = None
            for cand in ("mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0", "resnet18"):
                if cand in avail:
                    fallback = cand
                    break
            if fallback is None and avail:
                fallback = avail[0]
            if fallback is None:
                raise RuntimeError("No compatible timm encoder found. Please install a newer timm or choose a different model.")
            LOGGER.warning("Falling back to encoder '%s' due to unavailable requested encoders: %s", fallback, candidates)
            enc = _try_create_features_model(timm, fallback, in_chans=in_ch, pretrained=False)
            if enc is None:
                raise RuntimeError(f"Failed to initialize fallback encoder '{fallback}'.")
            chosen = (fallback, enc)

        enc_name, enc_mod = chosen
        if enc_name != enc_name_req:
            LOGGER.warning("Using encoder '%s' instead of requested '%s'", enc_name, enc_name_req)
        self.encoder = enc_mod

        feat_chs: List[int] = list(self.encoder.feature_info.channels())  # type: ignore[attr-defined]
        # Reduce channels for each feature map to a common width
        self.lateral = nn.ModuleList([Conv1x1(c, base) for c in feat_chs])
        # Top-down smoothing after fusion at each stage (except the last)
        self.smooth = nn.ModuleList([Conv3x3(base, base) for _ in range(len(feat_chs) - 1)])
        # Final head
        self.head = nn.Conv2d(base, num_classes, kernel_size=1)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"SegFormerLite initialized: encoder={enc_name}, params={n_params/1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = self.encoder(x)
        # Build top-down path starting from the deepest feature
        feats_lat = [lat(f) for f, lat in zip(feats, self.lateral)]
        y = feats_lat[-1]
        for i in range(len(feats_lat) - 2, -1, -1):
            y = torch.nn.functional.interpolate(y, size=feats_lat[i].shape[-2:], mode="bilinear", align_corners=False)
            y = y + feats_lat[i]
            y = self.smooth[i](y) if i < len(self.smooth) else y
        logits = self.head(y)
        # Upsample logits to input spatial size to match target masks
        if logits.shape[-2:] != x.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
