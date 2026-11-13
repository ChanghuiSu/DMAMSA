from .data_lld import LoadData
from .data_nih import LoadData as NIHData

__all__ = ["LoadData", "create_dataset"]


def create_dataset(cfg, split="train", source_only=False):
    """Select the configured dataset loader; optionally load only non-contrast images for inference."""
    dataset = cfg.get("dataset", "lld").lower()
    loaders = {"lld": LoadData, "duke": LoadData, "nih": NIHData}
    defaults = {"lld": ["C_pre_256", "C_A_256_registered"],
                "duke": ["C_pre_256", "C_A_256_registered"],
                "nih": ["PRE", "ART"]}
    if dataset not in loaders:
        raise ValueError(f"不支持的数据集：{dataset}")
    modalities = cfg.get("modalities", defaults[dataset])
    if len(modalities) != 2:
        raise ValueError("modalities 必须按顺序包含输入模态和目标模态")
    validation = split == "val"
    kwargs = {}
    if dataset == "nih":
        kwargs = {key: cfg[key] for key in ("target_size", "resize_size") if key in cfg}
    return loaders[dataset](
        cfg["dataroot"], modalities[:1] if source_only else modalities,
        cfg["data_list"],
        start_num=cfg["start_num_val" if validation else "start_num"],
        end_num=cfg["end_num_val" if validation else "end_num"],
        start_slice=cfg.get("start_slice", 0), end_slice=cfg.get("end_slice"),
        debugging=cfg.get("debugging", False), **kwargs)
