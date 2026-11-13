import torch.nn.functional as F

from .base import PairedVolumeDataset


class LoadData(PairedVolumeDataset):
    """Load MRISegmenter images, apply centered padding, and resize to the requested shape."""

    def __init__(self, root, modalities, data_list, start_num=0, end_num=None,
                 start_slice=0, end_slice=None, target_size=(320, 320),
                 resize_size=(256, 256), debugging=False):
        self.target_size = target_size
        self.resize_size = resize_size
        super().__init__(root, modalities, data_list, start_num, end_num,
                         start_slice, end_slice, debugging)

    def prepare_slice(self, image):
        if self.target_size:
            height, width = image.shape[-2:]
            target_h, target_w = self.target_size
            top = (target_h - height) // 2
            left = (target_w - width) // 2
            image = F.pad(image.unsqueeze(0), (left, target_w - width - left,
                                               top, target_h - height - top),
                          mode="reflect").squeeze(0)
        if self.resize_size:
            image = F.interpolate(image.unsqueeze(0), size=self.resize_size,
                                  mode="bilinear", align_corners=False).squeeze(0)
        return image
