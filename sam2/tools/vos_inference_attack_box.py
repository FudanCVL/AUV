import argparse
import os
import multiprocessing as mp

import numpy as np
import torch
from PIL import Image
import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2_video_predictor
def load_uap(path, device="cpu", eps=10.0 / 255.0):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    uap = state.get("uap") if isinstance(state, dict) else state
    if not isinstance(uap, torch.Tensor) or not uap.is_floating_point():
        raise ValueError(f"No floating-point UAP tensor found in {path}")
    if uap.ndim == 3:
        uap = uap.unsqueeze(0)
    if uap.ndim != 4 or uap.shape[:2] != (1, 3) or not torch.isfinite(uap).all():
        raise ValueError(f"Invalid UAP tensor in {path}")
    return uap.detach().clamp(-eps, eps).to(device)

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

def load_ann_png(path):
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette

def save_ann_png(path, mask, palette):
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)

def get_per_obj_mask(mask):
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask

def put_per_obj_mask(per_obj_mask, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask

def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette

def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

def compute_mask_bbox(mask):
    """
    Compute the bounding box of a binary mask.
    
    Inputs:
    - mask: numpy array of shape (H, W), dtype=bool or uint8
    
    Returns:
    - box: (x_min, y_min, x_max, y_max) coordinates as numpy array of shape (4,), dtype=float32
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    ys, xs = np.where(mask > 0)
    
    if len(xs) == 0 or len(ys) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()
    
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

def apply_uap_to_image(img_batch, uap, image_size=1024):
    B, C, H, W = img_batch.shape
    assert uap.size(0) == 1
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_batch.device).view(1, 3, 1, 1)
    img_01 = img_batch * std + mean
    uap = uap.expand(B, -1, -1, -1)
    img_01_perturbed = torch.clamp(img_01 + uap, 0, 1)
    perturbed_img = (img_01_perturbed - mean) / std
    return perturbed_img

def split_videos_by_gpu(video_names, num_gpus, gpu_id):
    total = len(video_names)
    videos_per_gpu = total // num_gpus
    remainder = total % num_gpus
    
    start_idx = gpu_id * videos_per_gpu + min(gpu_id, remainder)
    end_idx = start_idx + videos_per_gpu + (1 if gpu_id < remainder else 0)
    
    return video_names[start_idx:end_idx]

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference_attack(
    predictor,
    uap,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
):
    original_forward_image = predictor.forward_image
    
    def forward_image_with_uap(img_batch):
        perturbed_img = apply_uap_to_image(img_batch, uap, predictor.image_size)
        return original_forward_image(perturbed_img)
    
    predictor.forward_image = forward_image_with_uap
    
    try:
        video_dir = os.path.join(base_video_dir, video_name)
        frame_names = [
            os.path.splitext(p)[0]
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        inference_state = predictor.init_state(
            video_path=video_dir, async_loading_frames=False
        )
        height = inference_state["video_height"]
        width = inference_state["video_width"]
        input_palette = None

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
                    for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                    for idx, name in enumerate(frame_names)
                    if os.path.exists(
                        os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                    )
                ]
            if len(input_frame_inds) == 0:
                raise RuntimeError(
                    f"In {video_name=}, got no input masks in {input_mask_dir=}."
                )
            input_frame_inds = sorted(set(input_frame_inds))

        for input_frame_idx in input_frame_inds:
            object_ids_set = None
            try:
                per_obj_input_mask, input_palette = load_masks_from_dir(
                    input_mask_dir=input_mask_dir,
                    video_name=video_name,
                    frame_name=frame_names[input_frame_idx],
                    per_obj_png_file=per_obj_png_file,
                )
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}."
                ) from e
            if object_ids_set is None:
                object_ids_set = set(per_obj_input_mask)
                
            for object_id, object_mask in per_obj_input_mask.items():
                if object_id not in object_ids_set:
                    raise RuntimeError(
                        f"In {video_name=}, got a new {object_id=} appearing only in a "
                        f"later {input_frame_idx=}."
                    )
                bbox = compute_mask_bbox(object_mask)
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    box=bbox.reshape(1, 4),  # Shape: (1, 4) for single box
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
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx = input_frame_idx,
            ):
                per_obj_output_mask = {
                    out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                video_segments[out_frame_idx] = per_obj_output_mask

            for out_frame_idx, per_obj_output_mask in video_segments.items():
                save_masks_to_dir(
                    output_mask_dir=output_mask_dir,
                    video_name=video_name,
                    frame_name=frame_names[out_frame_idx],
                    per_obj_output_mask=per_obj_output_mask,
                    height=height,
                    width=width,
                    per_obj_png_file=per_obj_png_file,
                    output_palette=output_palette,
                )
    finally:
        predictor.forward_image = original_forward_image

def run_single_gpu(args, gpu_id):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    print(f"[GPU {gpu_id}] Using device: {device}")
    print(f"[GPU {gpu_id}] Loading UAP from {args.uap_checkpoint}")
    uap = load_uap(args.uap_checkpoint, device=device)
    
    print(f"[GPU {gpu_id}] UAP shape: {uap.shape}")
    print(f"[GPU {gpu_id}] UAP range: [{uap.min().item():.6f}, {uap.max().item():.6f}]")

    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=hydra_overrides_extra,
        vos_optimized=args.use_vos_optimized_video_predictor,
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
    
    all_video_names.sort()
    video_names = split_videos_by_gpu(all_video_names, args.num_gpus, gpu_id)
    print(f"[GPU {gpu_id}] Total videos: {len(all_video_names)}, assigned: {len(video_names)}")
    if len(video_names) > 5:
        print(f"[GPU {gpu_id}] Processing videos: {video_names[:5]}...")
    else:
        print(f"[GPU {gpu_id}] Processing videos: {video_names}")

    for n_video, video_name in enumerate(video_names):
        print(f"\n[GPU {gpu_id}] {n_video + 1}/{len(video_names)} - running attack on {video_name}")
        vos_inference_attack(
            predictor=predictor,
            uap=uap,
            base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir,
            output_mask_dir=args.output_mask_dir,
            video_name=video_name,
            score_thresh=args.score_thresh,
            use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file,
        )

    print(
        f"[GPU {gpu_id}] Completed VOS attack prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--uap_checkpoint",
        type=str,
        default="runs/sam2_base_plus/stage2/checkpoints/checkpoint.pt",
        help="path to the UAP checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
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
        help="whether to use all available PNG files in input_mask_dir",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks",
    )
    parser.add_argument(
        "--use_vos_optimized_video_predictor",
        action="store_true",
        help="whether to use vos optimized video predictor with all modules compiled",
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
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError("One or more inference workers failed")
        print("All GPU processes completed!")
    else:
        gpu_id = args.gpu_id if args.gpu_id is not None else 0
        run_single_gpu(args, gpu_id)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
