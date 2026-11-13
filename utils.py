from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import yaml


def load_config(path):
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须包含 YAML 映射")
    return config


class MetricsEvaluator:
    """Compute per-slice metrics in [-1, 1] without accumulating state between calls."""

    def __init__(self, device):
        self.psnr_metric = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=2.0, gaussian_kernel=False, kernel_size=7).to(device)

    @torch.no_grad()
    def compute(self, pred, target, max_psnr_value=100.0):
        if pred.ndim != 4 or pred.shape != target.shape or pred.shape[0] == 0:
            raise ValueError("指标输入必须为形状相同的非空 [B, C, H, W] 张量")
        values = []
        for pred_slice, target_slice in zip(pred, target):
            psnr = self.psnr_metric(pred_slice.unsqueeze(0), target_slice.unsqueeze(0))
            ssim = self.ssim_metric(pred_slice.unsqueeze(0), target_slice.unsqueeze(0))
            values.append((max_psnr_value if torch.isinf(psnr) else psnr.item(), ssim.item()))
            self.psnr_metric.reset()
            self.ssim_metric.reset()
        return tuple(np.mean(values, axis=0))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        if not 0 <= decay_start_epoch < n_epochs:
            raise ValueError("decay_epoch 必须大于等于 0，且小于 n_epochs")
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return max(0.0, 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)
                   / (self.n_epochs - self.decay_start_epoch))


def save_png(pred_out, save_path, name):
    image = pred_out.detach().cpu().numpy()
    if image.shape[:2] != (1, 1) or image.ndim != 4:
        raise ValueError("PNG 导出要求输入尺寸为 [1, 1, H, W]")
    image = np.clip((image[0, 0] + 1) / 2 * 255, 0, 255).astype(np.uint8)
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(save_path / f"{name}.png")
