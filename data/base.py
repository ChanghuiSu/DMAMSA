from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class PairedVolumeDataset(Dataset):
    """Load subject volumes and verify matching slice identifiers and shapes across modalities."""

    def __init__(self, root, modalities, data_list, start_num=0, end_num=None,
                 start_slice=0, end_slice=None, debugging=False):
        super().__init__()
        self.modalities = list(modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("模态列表不能为空或包含重复项")
        with open(data_list, encoding="utf-8") as file:
            subjects = [line.strip() for line in file if line.strip()]
        subjects = subjects[start_num:end_num]
        if not subjects:
            raise ValueError(f"病例列表为空，请检查 {data_list} 的病例索引范围")

        self.data = {modality: [] for modality in self.modalities}
        for subject in subjects:
            paired_slices = {}
            reference_shape = None
            for modality in self.modalities:
                path = self.volume_path(Path(root), subject, modality)
                volume = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
                if volume.ndim != 3 or not np.isfinite(volume).all():
                    raise ValueError(f"体数据必须是有限值组成的三维数组：{path}")
                if reference_shape is not None and volume.shape != reference_shape:
                    raise ValueError(f"病例 {subject} 的 {modality} 与参考模态体数据尺寸不匹配")
                reference_shape = volume.shape
                if not volume.size:
                    raise ValueError(f"体数据为空：{path}")
                low, high = float(volume.min()), float(volume.max())
                volume = volume.astype(np.float32)
                # Map constant volumes to -1 to avoid division by zero during normalization.
                volume = ((volume - low) / (high - low) * 2 - 1
                          if high > low else np.full_like(volume, -1))
                indices = range(*slice(start_slice, end_slice).indices(len(volume)))
                slices = []
                for index in indices:
                    image = torch.from_numpy(volume[index].copy()).unsqueeze(0)
                    slices.append([self.prepare_slice(image), f"{subject}_{index}"])
                if not slices:
                    raise ValueError(f"切片范围为空：{path}，共 {len(volume)} 层")
                paired_slices[modality] = slices

            reference = paired_slices[self.modalities[0]]
            for modality, slices in paired_slices.items():
                if ([item[1] for item in slices] != [item[1] for item in reference]
                        or any(a[0].shape != b[0].shape
                               for a, b in zip(slices, reference))):
                    raise ValueError(f"病例 {subject} 的 {modality} 与参考模态切片不匹配")

            remaining = 128 - len(self) if debugging else len(reference)
            for modality in self.modalities:
                self.data[modality].extend(paired_slices[modality][:remaining])
            if debugging and len(self) >= 128:
                break
        print(f"加载了 {len(self)} 个配对切片")

    def volume_path(self, root, subject, modality):
        return root / subject / f"{modality}.nii.gz"

    def prepare_slice(self, image):
        return image

    def __getitem__(self, index):
        return {modality: self.data[modality][index] for modality in self.modalities}

    def __len__(self):
        return len(self.data[self.modalities[0]])
