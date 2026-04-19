"""Minimal helpers for ``inference_demo.ipynb``: ``DenseMark`` and Part 3 eval utilities."""
from __future__ import annotations

import math
import os
from typing import Dict

import torch
import torch.nn.functional as F

from watermark_anything.data.transforms import get_transforms_segmentation
from watermark_anything.data.loader import get_dataloader_segmentation
from watermark_anything.data.metrics import get_masked_bit_mode
from watermark_anything.modules.ldpc import LDPCSystem
from watermark_anything.augmentation.geometric import (
    Rotate,
    Resize,
    Crop,
    Perspective,
    HorizontalFlip,
    Identity,
)
from watermark_anything.augmentation.valuemetric import (
    JPEG,
    GaussianBlur,
    MedianFilter,
    Brightness,
    Contrast,
    Saturation,
    Hue,
)
from watermark_anything.utils.inference_utils import load_model_from_checkpoint

# --- color overlay (Fig.1 Part 2) ---


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h = (h % 1.0) * 6.0
    hi = torch.remainder(torch.floor(h).long(), 6)
    f = h - torch.floor(h)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    out = torch.zeros(*h.shape, 3, device=h.device, dtype=h.dtype)
    cases = [
        lambda: torch.stack([v, t, p], dim=-1),
        lambda: torch.stack([q, v, p], dim=-1),
        lambda: torch.stack([p, v, t], dim=-1),
        lambda: torch.stack([p, q, v], dim=-1),
        lambda: torch.stack([t, p, v], dim=-1),
        lambda: torch.stack([v, p, q], dim=-1),
    ]
    for i in range(6):
        m = hi == i
        if m.any():
            out = torch.where(m.unsqueeze(-1), cases[i](), out)
    return out


def _palette_distinct_blue_red_first(k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    blue = torch.tensor([0.10, 0.28, 0.72], device=device, dtype=dtype)
    red = torch.tensor([0.72, 0.08, 0.20], device=device, dtype=dtype)
    if k <= 0:
        return blue.unsqueeze(0)[:0]
    if k == 1:
        return blue.unsqueeze(0)
    if k == 2:
        return torch.stack([blue, red], dim=0)
    hues = [((j - 2) * 0.6180339887498949 + 0.271) % 1.0 for j in range(2, k)]
    h = torch.tensor(hues, device=device, dtype=dtype)
    extra = _hsv_to_rgb(h, torch.full_like(h, 0.78), torch.full_like(h, 0.90))
    return torch.cat([blue.unsqueeze(0), red.unsqueeze(0), extra], dim=0)


def recolor_bit_active_blue_light_red(cba: torch.Tensor) -> torch.Tensor:
    B, _, H, W = cba.shape
    device, dtype = cba.device, cba.dtype
    out = torch.zeros_like(cba)
    for b in range(B):
        flat = cba[b].permute(1, 2, 0).reshape(-1, 3)
        active = flat.sum(dim=-1) > 1e-4
        if not active.any():
            continue
        sub = (flat[active] * 512.0).round() / 512.0
        uniq, inv = torch.unique(sub, dim=0, return_inverse=True)
        k = uniq.shape[0]
        key = uniq.sum(1) * 1000.0 + uniq[:, 0] * 100 + uniq[:, 1] * 10 + uniq[:, 2]
        rank = torch.empty(k, dtype=torch.long, device=device)
        rank[torch.argsort(key)] = torch.arange(k, device=device)
        pal = _palette_distinct_blue_red_first(k, device, dtype)
        colored = torch.zeros_like(flat)
        colored[active] = pal[rank[inv]]
        out[b] = colored.reshape(H, W, 3).permute(2, 0, 1)
    return out


# --- Part 3 ---


def generate_mask_from_ratio(images_encoded: torch.Tensor, mask_ratio: float = 1.0) -> torch.Tensor:
    B, C, H, W = images_encoded.shape
    total_area = H * W
    mask_area = int(total_area * mask_ratio)
    side = int(math.sqrt(mask_area))
    center_h = H // 2
    center_w = W // 2
    top = max(center_h - side // 2, 0)
    bottom = min(center_h + side // 2, H)
    left = max(center_w - side // 2, 0)
    right = min(center_w + side // 2, W)
    mask = torch.zeros((B, 1, H, W), dtype=torch.float32, device=images_encoded.device)
    mask[:, 0, top:bottom, left:right] = 1.0
    return mask


def transform_type(transform_name: str) -> str:
    geo_transforms = {"perspective", "horizontalflip", "rotate", "crop", "resize"}
    value_transforms = {
        "brightness",
        "contrast",
        "hue",
        "saturation",
        "gaussianblur",
        "medianfilter",
        "jpeg",
    }
    iden_transforms = {"identity"}
    name = transform_name.lower()
    if name in geo_transforms:
        return "geo"
    if name in value_transforms:
        return "value"
    if name in iden_transforms:
        return "iden"
    return "none"


def watermark_bit_accuracy(msgs_encoded: torch.Tensor, msgs_decoded: torch.Tensor) -> float:
    if msgs_decoded.size(-1) == 0:
        return 0.5
    return (msgs_encoded == msgs_decoded.gt(0.5)).float().nanmean().item()


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class DenseMark:
    def __init__(self, device: torch.device):
        print("Loading DenseMark model...")

        json_path = os.path.join(_REPO_ROOT, "checkpoints", "params.json")
        ckpt_path = os.path.join(_REPO_ROOT, "checkpoints", "bce_bg_crl_llr.pth")

        wm_model = load_model_from_checkpoint(json_path, ckpt_path).to(device).eval()

        self.wm_model = wm_model
        self.nbits = 32
        self.system_ldpc = LDPCSystem(max_iter=10, device=device)
        self.device = device

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict:
        batch_size = images.shape[0]
        wm_msg = torch.randint(0, 2, (batch_size, self.nbits)).float().to(self.device)
        codeword = self.system_ldpc.encode(wm_msg)
        outputs = self.wm_model.embed(images, codeword)
        encoded_images = outputs["imgs_w"]

        return {
            "images_encoded": encoded_images,
            "msgs_encoded": wm_msg,
            "codeword": codeword,
        }

    @torch.no_grad()
    def decode(self, images: torch.Tensor) -> dict:
        preds = self.wm_model.detect(images)["preds"]
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        mask_preds = (mask_preds > 0.5).float()
        bit_preds = preds[:, 1:, :, :]
        _, _, post_info_bits, active_mask = self.system_ldpc.decode(bit_preds)

        color_bit = self.bits_to_rgb(post_info_bits)
        color_bit_active = color_bit.float() * active_mask.float()

        unique_vectors = self.get_unique_32_vectors(post_info_bits * active_mask)
        if active_mask.dim() > 3 and active_mask.size(1) == 1:
            active_mask = active_mask.squeeze(1)

        ratio = active_mask.float().mean(dim=(1, 2))
        invalid = ratio <= 0.001
        active_mask[invalid] = True

        pred_message = get_masked_bit_mode(mask_preds, post_info_bits, active_mask)

        return {
            "msgs_decoded": pred_message,
            "masks_pred": mask_preds,
            "bit_preds": bit_preds,
            "post_info_bits": post_info_bits,
            "mult_bit": unique_vectors,
            "color_bit_active": color_bit_active,
        }

    @staticmethod
    def bits_to_rgb(bit_tensor: torch.Tensor) -> torch.Tensor:
        device = bit_tensor.device
        B, C, H, W = bit_tensor.shape

        powers = 2 ** torch.arange(C - 1, -1, -1, device=device, dtype=torch.long)
        powers = powers.view(1, C, 1, 1)

        values = (bit_tensor.long() * powers).sum(dim=1)

        flat_values = values.view(-1)
        uniq, inverse_idx = torch.unique(flat_values, return_inverse=True)

        num_colors = len(uniq)
        colors = torch.rand(num_colors, 3, device=device)

        color_mask = colors[inverse_idx]
        color_mask = color_mask.view(B, H, W, 3).permute(0, 3, 1, 2)

        return color_mask

    @staticmethod
    def get_unique_32_vectors(x: torch.Tensor, min_count: int = 200) -> list:
        assert x.dim() == 4 and x.size(1) == 32, f"Expected [B, 32, H, W], got {x.shape}"

        batch_size = x.size(0)
        all_filtered_vectors = []

        for b in range(batch_size):
            current_batch = x[b].permute(1, 2, 0).reshape(-1, 32)

            unique_vectors, counts = torch.unique(
                current_batch, dim=0, return_inverse=False, return_counts=True
            )

            count_mask = counts >= min_count
            non_zero_mask = unique_vectors.abs().sum(dim=1) > 0
            mask = count_mask & non_zero_mask

            all_filtered_vectors.append(unique_vectors[mask])

        return all_filtered_vectors


def repo_root() -> str:
    return _REPO_ROOT


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_validation_augs():
    return [
        (Identity, [0]),
        (Brightness, [1.5, 2.0]),
        (Contrast, [1.5, 2.0]),
        (Hue, [-0.1, 0.1]),
        (Saturation, [1.5, 2.0]),
        (GaussianBlur, [3, 17]),
        (MedianFilter, [3, 7]),
        (JPEG, [60, 80]),
        (HorizontalFlip, [0]),
        (Crop, [0.7, 0.85]),
        (Resize, [0.3, 0.5]),
        (Rotate, [-10, 10]),
        (Perspective, ["0.1_val", "0.3_val"]),
    ]


def build_wam_sun_val_loader(
    data_dir: str,
    val_subdir: str,
    ann_file: str,
    image_size: int,
    batch_size: int,
    workers: int,
):
    _, _, val_transform, val_mask_transform = get_transforms_segmentation(image_size)
    return get_dataloader_segmentation(
        data_dir=os.path.join(data_dir, val_subdir),
        ann_file=os.path.join(data_dir, "annotations", ann_file),
        transform=val_transform,
        mask_transform=val_mask_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        random_nb_object=False,
        multi_w=False,
        max_nb_masks=3,
    )


def wam_sun_mean_by_class(bit_acc_by_class: Dict[str, list]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for aug_cls in ("geo", "value", "iden"):
        vals = bit_acc_by_class.get(aug_cls, [])
        out[aug_cls] = sum(vals) / len(vals) if vals else float("nan")
    return out
