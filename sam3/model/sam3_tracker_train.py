"""
SAM3 Tracker for VOS training (similar to SAM2Train).
"""

import logging
import numpy as np
import torch
from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.sam3_tracker_utils import get_next_point, sample_box_points
from sam3.train.data.collator import BatchedDatapoint

logger = logging.getLogger(__name__)


class Sam3TrackerTrain(Sam3TrackerBase):
    def __init__(
        self,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        num_init_cond_frames_for_train=1,
        num_init_cond_frames_for_eval=1,
        rand_init_cond_frames_for_train=False,
        rand_init_cond_frames_for_eval=False,
        add_all_frames_to_correct_as_cond=False,
        num_correction_pt_per_frame=7,
        pt_sampling_for_eval="center",
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        freeze_image_encoder=False,
        prob_to_dropout_spatial_mem=0.0,
        teacher_force_obj_scores_for_mem=False,
        use_memory_selection=False,
        pretrained_uap=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval

        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval

        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling

        self.rng = np.random.default_rng(seed=42)
        self.prob_to_dropout_spatial_mem = prob_to_dropout_spatial_mem
        self.teacher_force_obj_scores_for_mem = teacher_force_obj_scores_for_mem

        self.use_memory_selection = use_memory_selection

        if freeze_image_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        image_size = getattr(self, "image_size", 1024)
        assert image_size == 1008, "image_size must be 1008"

        eps_pixel = float(10.0 / 255.0)
        self.uap_eps = eps_pixel
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean)
        self.register_buffer("img_std", std)
        self.uap = torch.nn.Parameter(
            torch.empty(1, 3, image_size, image_size).uniform_(-1.0, 1.0) * eps_pixel,
            requires_grad=True,
        )
        if pretrained_uap is not None:
            self._load_pretrained_uap(pretrained_uap)

        for name, p in self.named_parameters():
            if "uap" not in name:
                p.requires_grad = False

    def _load_pretrained_uap(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        uap = state.get("uap") if isinstance(state, dict) else state
        if not isinstance(uap, torch.Tensor) or not uap.is_floating_point():
            raise ValueError(f"No floating-point UAP tensor found in {path}")
        if uap.ndim == 3:
            uap = uap.unsqueeze(0)
        if uap.shape != self.uap.shape:
            raise ValueError(
                f"UAP shape {tuple(uap.shape)} does not match {tuple(self.uap.shape)}"
            )
        if not torch.isfinite(uap).all():
            raise ValueError(f"UAP contains non-finite values: {path}")
        with torch.no_grad():
            self.uap.copy_(uap.clamp(-self.uap_eps, self.uap_eps))
        logger.info("Loaded pretrained UAP from %s", path)

    def forward(self, input: BatchedDatapoint):
        if self.training or not self.forward_backbone_per_frame_for_eval:
            img_batch = input.img_batch
            img_01 = img_batch * self.img_std + self.img_mean
            with torch.no_grad():
                self.uap.data = torch.clamp(self.uap.data, -self.uap_eps, self.uap_eps)
            uap = self.uap.expand(img_01.shape[0], -1, -1, -1)
            img_01_adv = torch.clamp(img_01 + uap, 0, 1)
            img_adv = (img_01_adv - self.img_mean) / self.img_std
            backbone_out = self.forward_image(img_adv)

        else:
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        return previous_stages_out

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Prepare input mask, point or box prompts.
        """
        gt_masks_per_frame = {}
        for stage_id in range(len(input.find_inputs)):
            segments = input.find_targets[stage_id].segments
            gt_masks_per_frame[stage_id] = segments.unsqueeze(1)

        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = len(input.find_inputs)
        backbone_out["num_frames"] = num_frames

        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval

        assert num_init_cond_frames >= 1
        use_pt_input = self.rng.random() < prob_to_use_pt_input

        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            if rand_init_cond_frames:
                num_init_cond_frames = self.rng.integers(
                    1, num_init_cond_frames, endpoint=True
                )
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        backbone_out["use_pt_input"] = use_pt_input

        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        for t in init_cond_frames:
            if not use_pt_input:

                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                assert False

                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(gt_masks_per_frame[t])
                else:
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t],
                        pred_masks=None,
                        method=(
                            "uniform" if self.training else self.pt_sampling_for_eval
                        ),
                    )

                point_inputs = {"point_coords": points, "point_labels": labels}
                backbone_out["point_inputs_per_frame"][t] = point_inputs

        if not use_pt_input:
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            if rand_frames_to_correct and num_frames_to_correct > num_init_cond_frames:
                num_frames_to_correct = self.rng.integers(
                    num_init_cond_frames, num_frames_to_correct, endpoint=True
                )
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    backbone_out["frames_not_in_init_cond"], extra_num, replace=False
                ).tolist()
            )

        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt

        return backbone_out

    def back_convert(self, targets):
        """
        Convert targets for loss computation.
        For VOS, we return targets as-is since Sam3LossVOS
        expects BatchedFindTarget format.
        """
        return targets
