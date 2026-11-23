"""Clean/attacked language segmentation of assets/people.mp4.

Both videos show instance masks over the original frames for comparison.
Instance colors/IDs are stable within each run, not matched across runs.
"""
import argparse
import gc
import sys
from functools import partial
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sam3.model_builder import build_sam3_video_model
from sam3.eval.vos_inference_attack_mask import apply_uap_to_image, load_uap


def load_mp4_frames(resource_path, image_size, img_mean, img_std, **kwargs):
    """Normalize each frame in NumPy, avoiding the upstream MP4 loader.

    The upstream cv2 path omits /255 and materializes a full float32 clip.
    Store individual contiguous float16 CHW frames instead.
    """
    capture = cv2.VideoCapture(str(resource_path))
    frames = []
    mean = np.asarray(img_mean, dtype=np.float32)
    std = np.asarray(img_std, dtype=np.float32)
    height = width = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {resource_path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
            normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
            chw = np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float16)
            frames.append(torch.from_numpy(chw))
        if not frames:
            raise ValueError("Video contains no decodable frames")
        if total > 0 and len(frames) != total:
            raise ValueError(f"Decoded {len(frames)} frames, expected {total}")
        return frames, height, width
    finally:
        capture.release()


def draw_instances(frame, outputs):
    result = frame.copy()
    for obj_id, mask in zip(outputs["out_obj_ids"], outputs["out_binary_masks"]):
        mask = np.asarray(mask, dtype=bool).reshape(frame.shape[:2])
        if not mask.any():
            continue
        hsv = np.uint8([[[(int(obj_id) * 47) % 180, 200, 255]]])
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        result[mask] = (0.55 * result[mask] + 0.45 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, tuple(map(int, color)), 2)
        ys, xs = np.where(mask)
        cv2.putText(result, f"ID {int(obj_id)}", (int(xs.mean()), max(20, int(ys.min()))),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return result


@torch.inference_mode()
def run_video(model, video_path, destination, text):
    # Independent states prevent clean feature/memory leakage into attack.
    # Override only this process's loader during initialization; no SAM3 source
    # files are modified. The original loader is restored immediately afterward.
    with patch("sam3.model.sam3_video_inference.load_resource_as_video_frames", load_mp4_frames):
        state = model.init_state(resource_path=str(video_path), offload_video_to_cpu=True)
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS")
        size = (state["orig_width"], state["orig_height"])
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {destination}")
        model.add_prompt(inference_state=state, frame_idx=0, text_str=text)
        count = 0
        for frame_idx, outputs in model.propagate_in_video(
            inference_state=state, start_frame_idx=0,
            max_frame_num_to_track=state["num_frames"], reverse=False,
        ):
            if frame_idx != count:
                raise RuntimeError(f"Expected frame {count}, got {frame_idx}")
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_idx}")
            writer.write(draw_instances(frame, outputs))
            # This demo performs one forward pass, with no interactive edits or
            # backward propagation. Already-exported interactive output caches
            # are not needed by subsequent tracking steps.
            state["cached_frame_outputs"].pop(frame_idx, None)
            count += 1
        if count != state["num_frames"]:
            raise RuntimeError(f"Incomplete output: {count} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        del state
        gc.collect()
        torch.cuda.empty_cache()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", default="checkpoints/sam3.pt")
    parser.add_argument("--uap_checkpoint", required=True)
    parser.add_argument("--bpe_path", default=None)
    parser.add_argument("--text_prompt", default="person")
    parser.add_argument("--output_dir", default="output/sam3_language")
    args = parser.parse_args()
    # Avoid CPU oversubscription on shared multi-core servers.
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    video_path = Path("assets/people.mp4")
    for path in (video_path, Path(args.checkpoint_path), Path(args.uap_checkpoint)):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if not args.text_prompt.strip():
        parser.error("Text prompt must not be empty")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    output = Path(args.output_dir)
    for name in ("clean.mp4", "attacked.mp4"):
        if (output / name).exists():
            parser.error(f"Output exists: {output / name}; choose a fresh --output_dir")
    output.mkdir(parents=True, exist_ok=True)
    model = build_sam3_video_model(
        checkpoint_path=args.checkpoint_path, bpe_path=args.bpe_path,
        device="cuda:0", compile=False,
    ).eval()
    # Use the tracker's native CPU state offload to bound GPU memory for many
    # people over a long clip; keep the same model and detection parameters.
    model.tracker.init_state = partial(model.tracker.init_state, offload_state_to_cpu=True)
    uap = load_uap(args.uap_checkpoint, device="cuda:0")
    if tuple(uap.shape) != (1, 3, model.image_size, model.image_size):
        raise ValueError(f"Unexpected UAP shape: {tuple(uap.shape)}")
    if not torch.isfinite(uap).all():
        raise ValueError("UAP contains non-finite values")
    original_forward_image = model.detector.backbone.forward_image

    def forward_image_with_uap(samples):
        # Perturb shared detector/tracker image features once, on every frame.
        return original_forward_image(apply_uap_to_image(samples, uap))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        print("Clean propagation", flush=True)
        run_video(model, video_path, output / "clean.mp4", args.text_prompt)
        model.detector.backbone.forward_image = forward_image_with_uap
        try:
            print("Adversarial propagation", flush=True)
            run_video(model, video_path, output / "attacked.mp4", args.text_prompt)
        finally:
            model.detector.backbone.forward_image = original_forward_image


if __name__ == "__main__":
    main()
