import os
from PIL import Image as PILImage
import numpy as np
from torch.utils.data import Dataset
import torch

from .sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image,
    Object,
    InferenceMetadata,
)


class Sam3VOSDataset(Dataset):
    """
    VOS dataset that returns consecutive frames from the same video.
    Each sample is a video clip with N consecutive frames.
    """

    def __init__(
        self,
        root_dir: str,
        transforms=None,
        training: bool = True,
        multiplier: int = 1,
        max_ann_per_img: int = 200,
        num_frames_per_clip: int = 8,
        frame_stride: int = 1,
    ):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "JPEGImages")
        self.ann_dir = os.path.join(root_dir, "Annotations")
        self._transforms = transforms
        self.training = training
        self.multiplier = multiplier
        self.max_ann_per_img = max_ann_per_img
        self.curr_epoch = 0
        self.num_frames_per_clip = num_frames_per_clip
        self.frame_stride = frame_stride

        video_folders = sorted(
            [
                d
                for d in os.listdir(self.img_dir)
                if os.path.isdir(os.path.join(self.img_dir, d))
            ]
        )

        print(f"Found {len(video_folders)} video folders")

        self.videos = []
        for video_name in video_folders:
            video_img_dir = os.path.join(self.img_dir, video_name)
            video_ann_dir = os.path.join(self.ann_dir, video_name)
            frame_files = sorted(
                [f for f in os.listdir(video_img_dir) if f.endswith((".jpg", ".png"))]
            )
            if len(frame_files) < num_frames_per_clip:
                continue
            self.videos.append(
                {
                    "video_name": video_name,
                    "video_img_dir": video_img_dir,
                    "video_ann_dir": video_ann_dir,
                    "frame_files": frame_files,
                    "num_frames": len(frame_files),
                }
            )
        print(
            f"Found {len(self.videos)} valid videos (with >= {num_frames_per_clip} frames)"
        )
        self.videos = self.videos * multiplier
        print(f"After multiplier ({multiplier}): {len(self.videos)} samples")

    def __len__(self):

        return len(self.videos)

    def set_curr_epoch(self, epoch: int):
        self.curr_epoch = epoch

    def __getitem__(self, idx):
        dataset_idx = int(idx) % len(self.videos)
        n_videos = len(self.videos)
        N = self.num_frames_per_clip
        clip_frames = None

        for _video_trial in range(n_videos):
            video_info = self.videos[dataset_idx]
            frame_files = video_info["frame_files"]
            num_frames = video_info["num_frames"]

            stride_eff = self.frame_stride
            max_start_idx = num_frames - (N - 1) * stride_eff
            if max_start_idx <= 0:
                stride_eff = 1
                max_start_idx = num_frames - (N - 1) * stride_eff
            if max_start_idx <= 0:
                dataset_idx = (dataset_idx + 1) % n_videos
                continue

            for _attempt in range(100):
                start_idx = int(np.random.randint(0, max_start_idx))
                frame_indices = [start_idx + t * stride_eff for t in range(N)]

                valid_clip = True
                first_frame_ids = None
                trial_clip = []

                for t, frame_i in enumerate(frame_indices):
                    frame_name = frame_files[frame_i]
                    img_path = os.path.join(video_info["video_img_dir"], frame_name)
                    mask_path = os.path.join(
                        video_info["video_ann_dir"],
                        os.path.splitext(frame_name)[0] + ".png",
                    )

                    if not os.path.exists(mask_path):
                        valid_clip = False
                        break

                    mask = PILImage.open(mask_path)
                    if mask.mode == "RGB":
                        mask = mask.convert("L")
                    mask_np = np.array(mask)
                    current_frame_ids = set(np.unique(mask_np))
                    current_frame_ids.discard(0)

                    if t == 0:
                        first_frame_ids = current_frame_ids
                        if not first_frame_ids:
                            valid_clip = False
                            break
                    else:
                        if not current_frame_ids.issubset(first_frame_ids):
                            valid_clip = False
                            break

                    trial_clip.append(
                        {
                            "image_path": img_path,
                            "mask_path": mask_path,
                            "frame_name": frame_name,
                        }
                    )

                if valid_clip:
                    clip_frames = trial_clip
                    break

            if clip_frames is not None:
                break

            print(
                f"Warning: Failed to find valid clip for video {video_info['video_name']}, "
                f"try next entry"
            )
            dataset_idx = (dataset_idx + 1) % n_videos

        if clip_frames is None:
            raise RuntimeError(
                "No valid clip found after scanning all videos (check stride, clip length, masks)"
            )

        MAX_OBJECTS = 1

        first_frame_mask = PILImage.open(clip_frames[0]["mask_path"])
        if first_frame_mask.mode == "RGB":
            first_frame_mask = first_frame_mask.convert("L")
        first_frame_mask_np = np.array(first_frame_mask)
        first_frame_obj_ids = sorted(
            int(x) for x in np.unique(first_frame_mask_np) if x > 0
        )
        if len(first_frame_obj_ids) > MAX_OBJECTS:
            first_frame_obj_ids = list(
                np.random.choice(first_frame_obj_ids, size=MAX_OBJECTS, replace=False)
            )

        all_sam3_images = []
        all_raw_images = []

        for frame_info in clip_frames:
            pil_image = PILImage.open(frame_info["image_path"]).convert("RGB")
            img_width, img_height = pil_image.size

            mask = PILImage.open(frame_info["mask_path"])
            if mask.mode == "RGB":
                mask = mask.convert("L")
            mask_np = np.array(mask)

            objects = []
            for obj_idx, obj_id in enumerate(first_frame_obj_ids):
                if obj_id in mask_np:
                    obj_mask = (mask_np == obj_id).astype(np.uint8)
                else:
                    obj_mask = np.zeros((img_height, img_width), dtype=np.uint8)

                obj = Object(
                    bbox=torch.tensor([0.0, 0.0, float(img_width), float(img_height)]),
                    area=1.0,
                    object_id=obj_idx,
                    frame_index=len(all_sam3_images),
                    segment=torch.from_numpy(obj_mask),
                    is_crowd=False,
                    source="vos",
                )
                objects.append(obj)

            sam3_image = Image(
                data=pil_image,
                objects=objects,
                size=(img_height, img_width),
            )
            all_sam3_images.append(sam3_image)
            all_raw_images.append(pil_image)

        find_queries = []
        for frame_idx in range(len(clip_frames)):
            frame_objects = all_sam3_images[frame_idx].objects
            object_ids = [obj.object_id for obj in frame_objects]

            find_query = FindQueryLoaded(
                query_text="",
                image_id=frame_idx,
                object_ids_output=object_ids,
                is_exhaustive=True,
                query_processing_order=frame_idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=dataset_idx,
                    original_image_id=dataset_idx,
                    original_category_id=1,
                    original_size=(img_height, img_width),
                    object_id=0,
                    frame_index=frame_idx,
                ),
            )
            find_queries.append(find_query)

        datapoint = Datapoint(
            find_queries=find_queries,
            images=all_sam3_images,
            raw_images=all_raw_images,
        )

        if self._transforms is not None:
            for transform in self._transforms:
                datapoint = transform(datapoint, epoch=self.curr_epoch)

        return datapoint
