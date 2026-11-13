import argparse
import datetime
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import create_dataset
from model import Discriminator, TLP, weights_init_normal
from utils import LambdaLR, MetricsEvaluator, load_config, save_png


def get_device(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu_id", "0"))
    if cfg.get("use_gpu", True):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；如需使用 CPU，请设置 common.use_gpu: false")
        return torch.device("cuda")
    return torch.device("cpu")


def build_generator(cfg):
    return TLP(
        inp_A_channels=cfg.get("channels", 1),
        out_channels=cfg.get("channels", 1),
        num_refinement_blocks=cfg.get("num_refinement_blocks", 2),
        num_classes=cfg.get("num_classes", 2),
        att_kernel=cfg.get("att_kernel", 11),
    )


def check_batch(image, cfg):
    expected = (cfg.get("channels", 1), cfg["img_height"], cfg["img_width"])
    if tuple(image.shape[1:]) != expected:
        raise ValueError(f"图像尺寸 {tuple(image.shape[1:])} 与配置 {expected} 不一致")


def report_complexity(generator, discriminator, cfg, device):
    from thop import clever_format, profile

    image = torch.zeros(1, cfg["channels"], cfg["img_height"], cfg["img_width"], device=device)
    # Profile a copy to keep hooks and batch-normalization updates from affecting training or checkpoints.
    for name, model, inputs in (
        ("Generator", generator, (image, image)),
        ("Discriminator", discriminator, (image,)),
    ):
        with torch.no_grad():
            macs, params = profile(deepcopy(model).eval(), inputs=inputs, verbose=False)
        macs_text, params_text = clever_format([macs, params], "%.3f")
        print(f"{name}: MACs={macs_text}, parameters={params_text}")
    print("THOP 仅统计已支持的算子；注意力中的 einsum 等运算可能未被计入。")


def train_model(cfg):
    if cfg.get("epoch", 0) != 0:
        raise ValueError("当前入口不恢复优化器状态；新训练请设置 epoch: 0")
    device = get_device(cfg)
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "log.txt"
    with log_path.open("w", encoding="utf-8") as file:
        file.write(f"{cfg}\nstage epoch psnr ssim\n")

    train_set = create_dataset(cfg)
    val_set = create_dataset(cfg, split="val")
    dataloader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                            num_workers=cfg["n_cpu"], drop_last=True)
    val_data = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=cfg["n_cpu"])
    if not len(dataloader):
        raise ValueError("训练切片数小于 batch_size，请减小 batch_size")
    source, target = train_set.modalities
    check_batch(train_set[0][source][0].unsqueeze(0), cfg)
    check_batch(val_set[0][source][0].unsqueeze(0), cfg)

    generator = build_generator(cfg).to(device)
    discriminator = Discriminator(cfg["channels"]).to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    if cfg.get("profile", False):
        report_complexity(generator, discriminator, cfg, device)

    criterion_gan = nn.MSELoss()
    criterion_reconstruction = nn.L1Loss()
    evaluator = MetricsEvaluator(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=cfg["lr"], betas=(cfg["b1"], cfg["b2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=cfg["lr"], betas=(cfg["b1"], cfg["b2"]))
    decay = LambdaLR(cfg["n_epochs"], 0, cfg["decay_epoch"])
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=decay.step)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=decay.step)

    previous_time = time.time()
    for epoch in range(cfg["n_epochs"]):
        generator.train()
        discriminator.train()
        train_metrics = []
        for index, batch in enumerate(dataloader):
            input_image = batch[source][0].to(device)
            target_image = batch[target][0].to(device)
            optimizer_g.zero_grad()
            prediction = generator(input_image, target_image)

            optimizer_d.zero_grad()
            real_score = discriminator(target_image)
            fake_score = discriminator(prediction.detach())
            loss_d = (criterion_gan(real_score, torch.ones_like(real_score))
                      + criterion_gan(fake_score, torch.zeros_like(fake_score))) / 2
            loss_d.backward()
            optimizer_d.step()

            # Skip discriminator parameter gradients during the generator update.
            discriminator.requires_grad_(False)
            fake_score = discriminator(prediction)
            loss_reconstruction = criterion_reconstruction(prediction, target_image)
            loss_g = (criterion_gan(fake_score, torch.ones_like(fake_score))
                      + cfg["lambda_id"] * loss_reconstruction)
            loss_g.backward()
            optimizer_g.step()
            discriminator.requires_grad_(True)
            psnr, ssim = evaluator.compute(prediction.detach(), target_image)
            train_metrics.append((psnr, ssim))

            done = epoch * len(dataloader) + index + 1
            remaining = cfg["n_epochs"] * len(dataloader) - done
            eta = datetime.timedelta(seconds=int(remaining * (time.time() - previous_time)))
            previous_time = time.time()
            sys.stdout.write(
                f"\r[Epoch {epoch + 1}/{cfg['n_epochs']}] [Batch {index + 1}/{len(dataloader)}] "
                f"D: {loss_d.item():.4f} G: {loss_g.item():.4f} "
                f"PSNR: {psnr:.3f} SSIM: {ssim:.3f} ETA: {eta}"
            )
            sys.stdout.flush()

        scheduler_g.step()
        scheduler_d.step()
        torch.save(generator.state_dict(), ckpt_dir / "G_latest.pth")
        torch.save(discriminator.state_dict(), ckpt_dir / "D_latest.pth")
        interval = cfg.get("checkpoint_interval", 1)
        if interval > 0 and (epoch + 1) % interval == 0:
            torch.save(generator.state_dict(), ckpt_dir / f"G_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), ckpt_dir / f"D_{epoch + 1}.pth")

        generator.eval()
        val_metrics = []
        with torch.no_grad():
            for batch in val_data:
                input_image = batch[source][0].to(device)
                target_image = batch[target][0].to(device)
                # Evaluate each candidate separately, then average metrics over slices and branches.
                for branch in range(generator.phase_classifier.num_classes):
                    prediction = generator(input_image, phase_index=branch)
                    val_metrics.append(evaluator.compute(prediction, target_image))
        for stage, values in (("Training", train_metrics), ("Validation", val_metrics)):
            psnr, ssim = np.mean(values, axis=0)
            print(f"\n{stage}: PSNR={psnr:.3f}, SSIM={ssim:.3f}")
            with log_path.open("a", encoding="utf-8") as file:
                file.write(f"{stage} {epoch + 1} {psnr:.3f} {ssim:.3f}\n")
        previous_time = time.time()


def test_model(cfg):
    device = get_device(cfg)
    dataset = create_dataset(cfg, split="test", source_only=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    source = dataset.modalities[0]
    check_batch(dataset[0][source][0].unsqueeze(0), cfg)
    generator = build_generator(cfg).to(device)
    epoch = cfg.get("which_epoch", "latest")
    checkpoint = Path(cfg["ckpt_dir"]) / f"G_{epoch}.pth"
    generator.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    generator.eval()

    branch = cfg.get("phase_index", "all")
    branches = range(generator.phase_classifier.num_classes) if branch == "all" else [branch]
    if any(not isinstance(i, int) or not 0 <= i < generator.phase_classifier.num_classes for i in branches):
        raise ValueError("phase_index 应为 all 或有效的整数分支编号")
    output_dir = Path(cfg["ckpt_dir"]) / f"epoch_{epoch}{cfg.get('notes', '')}"
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            input_image = batch[source][0].to(device)
            filename = batch[source][1][0]
            for branch in branches:
                prediction = generator(input_image, phase_index=branch)
                save_png(prediction, output_dir / f"branch_{branch}", filename)
            print(f"\r[{index + 1}/{len(dataloader)}] {filename}", end="", flush=True)
    print(f"\n结果已保存到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DMAMSA-Net 训练与候选动脉期图像生成")
    parser.add_argument("--config", default="configs/config_lld.yaml", help="YAML 配置文件路径")
    parser.add_argument("--mode", choices=("train", "test"), help="覆盖配置中的运行模式")
    args = parser.parse_args()
    config = load_config(args.config)
    mode = args.mode or config.get("mode", "test")
    if mode not in ("train", "test"):
        raise ValueError(f"不支持的运行模式：{mode}")
    cfg = {**config["common"], **config[mode]}
    if mode == "train":
        train_model(cfg)
    else:
        test_model(cfg)


if __name__ == "__main__":
    main()
