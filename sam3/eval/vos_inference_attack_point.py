# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import argparse
import os
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sam3.model_builder import build_sam3_video_model

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def compute_mask_center_point(mask):
    """
    Compute the center point of a binary mask (centroid of all foreground pixels).
    Inputs: mask (H, W), dtype=bool or uint8
    Returns: (x, y) as np.array of shape (2,), dtype=float32
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        return np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0], dtype=np.float32)
    x_center = np.mean(coords[:, 1])
    y_center = np.mean(coords[:, 0])
    return np.array([x_center, y_center], dtype=np.float32)


def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_uap(checkpoint_path, device="cuda"):
    """Load UAP from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model" in checkpoint and "uap" in checkpoint["model"]:
        uap = checkpoint["model"]["uap"]
    elif "uap" in checkpoint:
        uap = checkpoint["uap"]
    else:

        uap = checkpoint

    if uap.dim() == 3:
        uap = uap.unsqueeze(0)
    uap = uap.to(device)
    uap.requires_grad_(False)
    uap = uap.clamp(-10.0 / 255.0, 10.0 / 255.0)
    return uap


def apply_uap_to_image(img_batch, uap, image_size=1024):
    """
    Apply UAP to image batch using SAM3 normalization.
    SAM3 uses mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
    """
    B, C, H, W = img_batch.shape
    assert uap.size(0) == 1
    mean = torch.tensor([0.5, 0.5, 0.5], device=img_batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=img_batch.device).view(1, 3, 1, 1)
    img_01 = img_batch * std + mean

    uap = uap.expand(B, -1, -1, -1)

    img_01_perturbed = torch.clamp(img_01 + uap, 0, 1)

    perturbed_img = (img_01_perturbed - mean) / std
    return perturbed_img


def split_videos_by_gpu(video_names, num_gpus, gpu_id):
    """Split video list across GPUs."""
    total = len(video_names)
    videos_per_gpu = total // num_gpus
    remainder = total % num_gpus

    start_idx = gpu_id * videos_per_gpu + min(gpu_id, remainder)
    end_idx = start_idx + videos_per_gpu + (1 if gpu_id < remainder else 0)

    return video_names[start_idx:end_idx]


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    uap=None,
):
    """Run VOS inference on a single video with SAM3 tracker predictor."""

    original_forward_image = None
    if uap is not None:
        original_forward_image = predictor.forward_image

        def forward_image_with_uap(img_batch):
            perturbed_img = apply_uap_to_image(img_batch, uap, predictor.image_size)
            return original_forward_image(perturbed_img)

        predictor.forward_image = forward_image_with_uap

    try:

        video_dir = os.path.join(base_video_dir, video_name)

        inference_state = predictor.init_state(video_path=video_dir)

        height = inference_state["video_height"]
        width = inference_state["video_width"]
        num_frames = inference_state["num_frames"]
        if num_frames > 1200:
            print(f"Skipping {video_name} as it has {num_frames} frames (limit: 1200).")
            return
        input_palette = None

        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(p))

        if len(frame_names) > 1500:
            print(
                f"Skipping {video_name} as it has {len(frame_names)} frames (limit: 1500)."
            )
            return

        if not use_all_masks:
            input_frame_inds = [0]
        else:
            if not per_obj_png_file:
                input_frame_inds = [
                    idx
                    for idx, name in enumerate(frame_names)
                    if os.path.exists(
                        os.path.join(input_mask_dir, video_name, f"{name}.png")
                    )
                ]
            else:
                input_frame_inds = [
                    idx
                    for object_name in os.listdir(
                        os.path.join(input_mask_dir, video_name)
                    )
                    for idx, name in enumerate(frame_names)
                    if os.path.exists(
                        os.path.join(
                            input_mask_dir, video_name, object_name, f"{name}.png"
                        )
                    )
                ]
            input_frame_inds = sorted(set(input_frame_inds))

        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}."
            )

        object_ids_set = None
        for input_frame_idx in input_frame_inds:
            input_mask_path = os.path.join(
                input_mask_dir, video_name, f"{frame_names[input_frame_idx]}.png"
            )
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask = get_per_obj_mask(input_mask)
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask.keys())
            for object_id, object_mask in per_obj_input_mask.items():
                center_point = compute_mask_center_point(object_mask)
                rel_center_point = np.array(
                    [[center_point[0] / width, center_point[1] / height]],
                    dtype=np.float32,
                )
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    points=rel_center_point,
                    labels=np.array([1], dtype=np.int32),
                    clear_old_points=True,
                    normalize_coords=True,
                )

        if object_ids_set is None or len(object_ids_set) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no object ids on {input_frame_inds=}."
            )

        os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
        output_palette = input_palette or DAVIS_PALETTE
        video_segments = {}

        for (
            frame_idx,
            obj_ids,
            low_res_masks,
            video_res_masks,
            obj_scores,
        ) in predictor.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=min(input_frame_inds),
            max_frame_num_to_track=num_frames,
            reverse=False,
            propagate_preflight=True,
        ):

            per_obj_output_mask = {}
            for i, obj_id in enumerate(obj_ids):

                if video_res_masks.dim() == 4:

                    mask = video_res_masks[i, 0]
                else:

                    mask = video_res_masks[i]

                mask_np = (mask > score_thresh).cpu().numpy().astype(bool)
                per_obj_output_mask[int(obj_id)] = mask_np

            video_segments[frame_idx] = per_obj_output_mask

        for frame_idx in range(num_frames):
            if frame_idx in video_segments:
                per_obj_output_mask = video_segments[frame_idx]
            else:
                per_obj_output_mask = {}

            output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, f"{frame_names[frame_idx]}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

    finally:

        if original_forward_image is not None:
            predictor.forward_image = original_forward_image


def run_single_gpu(args, gpu_id):
    """Run inference on a single GPU."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"[GPU {gpu_id}] Using device: {device}")

    if args.checkpoint_path:
        sam3_model = build_sam3_video_model(
            checkpoint_path=args.checkpoint_path,
            bpe_path=args.bpe_path,
        )
    else:
        sam3_model = build_sam3_video_model(bpe_path=args.bpe_path)

    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    uap = None
    if args.uap_checkpoint is not None:
        print(f"[GPU {gpu_id}] Loading UAP from {args.uap_checkpoint}")
        uap = load_uap(args.uap_checkpoint, device=device)
        print(f"[GPU {gpu_id}] UAP shape: {uap.shape}")
        print(
            f"[GPU {gpu_id}] UAP range: [{uap.min().item():.6f}, {uap.max().item():.6f}]"
        )

    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            all_video_names = [v.strip() for v in f.readlines()]
    else:
        all_video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]

    video_names = split_videos_by_gpu(all_video_names, args.num_gpus, gpu_id)
    print(
        f"[GPU {gpu_id}] Total videos: {len(all_video_names)}, assigned: {len(video_names)}"
    )
    if len(video_names) > 5:
        print(f"[GPU {gpu_id}] Processing videos: {video_names[:5]}...")
    else:
        print(f"[GPU {gpu_id}] Processing videos: {video_names}")

    if uap is not None:
        print(f"[GPU {gpu_id}] UAP attack is enabled")

    for n_video, video_name in enumerate(video_names):

        print(
            f"\n[GPU {gpu_id}] {n_video + 1}/{len(video_names)} - running on {video_name}"
        )
        try:
            vos_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
                uap=uap,
            )
        except Exception as e:
            print(f"[GPU {gpu_id}] Error processing {video_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(
        f"[GPU {gpu_id}] Completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="path to the SAM3 model checkpoint (optional, uses default if not provided)",
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default="./assets/bpe_simple_vocab_16e6.txt.gz",
        help="path to BPE vocabulary file",
    )
    parser.add_argument(
        "--uap_checkpoint",
        type=str,
        default=None,
        help="path to the UAP checkpoint (optional, if provided will apply UAP attack)",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files)",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files)",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available masks as input",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether to use separate per-object PNG files",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="total number of GPUs to use",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU ID for this process (0-indexed). If None and num_gpus>1, will auto-launch multiple processes",
    )
    args = parser.parse_args()

    if args.num_gpus > 1 and args.gpu_id is None:
        print(f"Auto-launching {args.num_gpus} GPU processes...")
        processes = []
        for gpu_id in range(args.num_gpus):
            p = mp.Process(target=run_single_gpu, args=(args, gpu_id))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        print("All GPU processes completed!")
    else:
        gpu_id = args.gpu_id if args.gpu_id is not None else 0
        run_single_gpu(args, gpu_id)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
