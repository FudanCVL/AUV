"""SAM3 clip sampling and resize/pad augmentation using SAM2 data structures."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.dataset.transforms import pad
from training.utils.data_utils import Frame, Object, VideoDatapoint


class Sam2VOSDataset(Dataset):
    def __init__(self, root_dir, transforms, num_frames_per_clip, frame_stride=1):
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.num_frames = num_frames_per_clip
        self.frame_stride = frame_stride
        self.epoch = 0
        self.videos = []
        # Match SAM3: sorted videos, random consecutive clip, one object selected
        # from the clip's first frame; do not introduce objects in later frames.
        for directory in sorted((self.root_dir / "JPEGImages").iterdir()):
            if not directory.is_dir():
                continue
            names = sorted(
                p.name for p in directory.iterdir() if p.suffix in (".jpg", ".png")
            )
            if len(names) >= self.num_frames:
                self.videos.append((directory.name, names))
        if not self.videos:
            raise ValueError(
                f"No videos with >= {self.num_frames} frames in {root_dir}"
            )
        print(f"Found {len(self.videos)} valid videos ({self.num_frames} frames/clip)")

    def __len__(self):
        return len(self.videos)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        for offset in range(len(self.videos)):
            video_id = (int(index) + offset) % len(self.videos)
            name, names = self.videos[video_id]
            stride = self.frame_stride
            max_start = len(names) - (self.num_frames - 1) * stride
            if max_start <= 0:
                stride = 1
                max_start = len(names) - self.num_frames + 1
            for _ in range(100):
                start = int(np.random.randint(max_start))
                indices = [start + t * stride for t in range(self.num_frames)]
                masks = []
                first_ids = set()
                for t, frame_idx in enumerate(indices):
                    mask_path = (
                        self.root_dir
                        / "Annotations"
                        / name
                        / (Path(names[frame_idx]).stem + ".png")
                    )
                    if not mask_path.is_file():
                        break
                    with Image.open(mask_path) as mask:
                        mask_array = np.array(
                            mask.convert("L") if mask.mode == "RGB" else mask
                        )
                    ids = set(np.unique(mask_array)) - {0}
                    if t == 0:
                        first_ids = ids
                        if not first_ids:
                            break
                    elif not ids.issubset(first_ids):
                        break
                    masks.append(mask_array)
                if len(masks) == self.num_frames:
                    return self._make_datapoint(
                        video_id, name, names, indices, masks, first_ids
                    )
        raise RuntimeError(
            "No valid clip found: check frames, annotations and object IDs"
        )

    def _make_datapoint(self, video_id, name, names, indices, masks, first_ids):
        ids = sorted(first_ids)
        object_id = (
            int(np.random.choice(ids, size=1, replace=False)[0])
            if len(ids) > 1
            else int(ids[0])
        )
        frames = []
        for t, (frame_idx, mask) in enumerate(zip(indices, masks)):
            with Image.open(
                self.root_dir / "JPEGImages" / name / names[frame_idx]
            ) as image:
                rgb = image.convert("RGB")
            width, height = rgb.size
            obj = Object(
                object_id=0,
                frame_index=t,
                segment=torch.from_numpy((mask == object_id).astype(np.uint8)),
            )
            frame = Frame(data=rgb, objects=[obj])
            frame.size = (height, width)
            frames.append(frame)
        datapoint = VideoDatapoint(
            frames=frames, video_id=video_id, size=(height, width)
        )
        for transform in self.transforms:
            datapoint = transform(datapoint, epoch=self.epoch)
        return datapoint


def resize_scales(size, min_size=480):
    return list(range(32 * ((min_size + 31) // 32), size + 1, 32))


class PadToSize:
    """Random placement in the model input square, independently per frame."""

    def __init__(self, size):
        self.size = size

    def __call__(self, datapoint, **kwargs):
        for i, frame in enumerate(datapoint.frames):
            width, height = frame.data.size
            dx, dy = self.size - width, self.size - height
            if min(dx, dy) < 0:
                raise ValueError("Resize output exceeds padding size")
            left, top = random.randint(0, dx), random.randint(0, dy)
            datapoint = pad(datapoint, i, (left, top, dx - left, dy - top))
        return datapoint
