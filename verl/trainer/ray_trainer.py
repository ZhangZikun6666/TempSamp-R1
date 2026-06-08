# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import importlib
import json
import math
import os
import time
import uuid
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt, thin_out_old_ckpts
from ..utils.logger import Tracker
from ..utils.multimodal_contract import validate_multi_modal_data_contract
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


def _disable_tqdm() -> bool:
    return os.getenv("VERL_DISABLE_TQDM", "0") == "1"


def _print_step_summary_enabled() -> bool:
    return os.getenv("VERL_PRINT_STEP_SUMMARY", "1") == "1"


def _skip_old_log_probs_enabled() -> bool:
    return os.getenv("VERL_SKIP_OLD_LOGPROBS", "0") == "1"


class _NoOpProgress:
    def update(self, *args, **kwargs) -> None:
        pass


def _fmt_metric(value: Any, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def _load_dotted_callable(spec: str):
    """Resolve a 'pkg.mod:attr' string into the callable it points to."""
    if ":" not in spec:
        raise ValueError(
            f"Expected a 'module.path:function' style spec, got {spec!r}. "
            "Example: 'examples.timelens.timelens_reward:build_gt_response'."
        )
    module_path, attr = spec.split(":", 1)
    module = importlib.import_module(module_path)
    try:
        fn = getattr(module, attr)
    except AttributeError as err:
        raise AttributeError(f"Module {module_path!r} has no attribute {attr!r}.") from err
    if not callable(fn):
        raise TypeError(f"{spec!r} resolved to a non-callable object: {type(fn).__name__}")
    return fn


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@torch.no_grad()
def compute_gtpo_advantage_diagnostics(
    data: DataProto,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Per-step GT-injection diagnostics for GTPO.

    Returns ``gtpo/*`` metrics that quantify how much GT injection shifts the
    GRPO group baseline (μ_op vs μ_g) and how the GT row's advantage compares
    to on-policy rows. No-op when no GT row is present (returns {}).
    """
    is_gt_row_np = data.non_tensor_batch.get("is_gt_row", None)
    if is_gt_row_np is None:
        return {}

    response_mask = data.batch["response_mask"]
    token_level_rewards = data.batch["token_level_rewards"]
    advantages = data.batch["advantages"]

    scores = (token_level_rewards * response_mask).sum(dim=-1)  # (bsz,)
    mask_sum = response_mask.sum(dim=-1).clamp(min=1)
    adv_signed = (advantages * response_mask).sum(dim=-1) / mask_sum  # (bsz,)

    bsz = scores.shape[0]
    is_gt = torch.tensor(
        [bool(is_gt_row_np[i]) for i in range(bsz)],
        dtype=torch.bool, device=scores.device,
    )

    uids = data.non_tensor_batch["uid"]
    uid_to_idx: dict[Any, list[int]] = defaultdict(list)
    for i, uid in enumerate(uids):
        uid_to_idx[uid].append(i)

    mu_op_acc, mu_g_acc = [], []
    sigma_op_acc, sigma_g_acc = [], []
    # Robust ratio aggregation: collect ratios ONLY from groups with non-trivial
    # sigma_op (above SIGMA_FLOOR), so a degenerate group with σ_op≈0 doesn't
    # blow up the per-batch average. Also report the unweighted mean(σ_g) /
    # mean(σ_op) as a stable companion.
    SIGMA_FLOOR = 1e-3
    contam_acc, sigma_ratio_acc = [], []
    degenerate_op_groups = 0
    sign_flip = 0
    total_op = 0
    no_op_pos_groups = 0
    n_groups = 0
    gt_mags, op_pos_mags, op_neg_mags = [], [], []

    for indices in uid_to_idx.values():
        if len(indices) < 2:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long, device=scores.device)
        g_scores = scores.index_select(0, idx_t)
        g_adv = adv_signed.index_select(0, idx_t)
        g_gt_mask = is_gt.index_select(0, idx_t)
        op_mask = ~g_gt_mask

        if op_mask.sum() < 1:
            continue

        op_scores = g_scores[op_mask]
        gt_scores = g_scores[g_gt_mask]

        mu_op = op_scores.mean()
        mu_g = g_scores.mean()
        sigma_op = (op_scores.std(unbiased=False)
                    if op_scores.numel() > 1 else torch.zeros_like(mu_op))
        sigma_g = g_scores.std(unbiased=False)

        mu_op_acc.append(mu_op.item())
        mu_g_acc.append(mu_g.item())
        sigma_op_v = float(sigma_op.item())
        sigma_g_v = float(sigma_g.item())
        sigma_op_acc.append(sigma_op_v)
        sigma_g_acc.append(sigma_g_v)
        if sigma_op_v < SIGMA_FLOOR:
            # σ_op too tiny → ratio/contamination divergent; skip from
            # per-group ratio averages but still count it.
            degenerate_op_groups += 1
        else:
            contam_acc.append(abs(float(mu_g.item()) - float(mu_op.item())) / sigma_op_v)
            sigma_ratio_acc.append(sigma_g_v / sigma_op_v)

        # Hypothetical baseline advantages (always full-group μ, σ) for the
        # on-policy rows — used to count sign-flips vs the actual variant.
        baseline_op_adv = (op_scores - mu_g) / (sigma_g + eps)
        actual_op_adv = g_adv[op_mask]

        sign_flips_mask = (
            ((actual_op_adv > 0) & (baseline_op_adv < 0))
            | ((actual_op_adv < 0) & (baseline_op_adv > 0))
        )
        sign_flip += int(sign_flips_mask.sum().item())
        total_op += int(op_mask.sum().item())

        if (baseline_op_adv <= 0).all():
            no_op_pos_groups += 1

        op_abs = actual_op_adv.abs()
        op_pos_mags.extend(op_abs[actual_op_adv > 0].tolist())
        op_neg_mags.extend(op_abs[actual_op_adv < 0].tolist())

        if g_gt_mask.any():
            gt_mags.extend(g_adv[g_gt_mask].abs().tolist())

        n_groups += 1

    if n_groups == 0:
        return {}

    mean_sigma_op = sum(sigma_op_acc) / n_groups if sigma_op_acc else 0.0
    mean_sigma_g = sum(sigma_g_acc) / n_groups if sigma_g_acc else 0.0
    mean_mu_op = sum(mu_op_acc) / n_groups if mu_op_acc else 0.0
    mean_mu_g = sum(mu_g_acc) / n_groups if mu_g_acc else 0.0

    out: dict[str, float] = {
        "gtpo/mu_op_mean": mean_mu_op,
        "gtpo/mu_g_mean": mean_mu_g,
        "gtpo/sigma_op_mean": mean_sigma_op,
        "gtpo/sigma_g_mean": mean_sigma_g,
        # Two ratio summaries — different aggregation:
        #  - *_per_group: average of per-group ratios (more sensitive but
        #    requires non-degenerate σ_op; we already filter < SIGMA_FLOOR).
        #  - *_of_means:  ratio of per-batch averages (always finite, more
        #    stable for swanlab plots; use this if per_group is noisy).
        "gtpo/sigma_ratio_per_group": (
            sum(sigma_ratio_acc) / max(len(sigma_ratio_acc), 1)
            if sigma_ratio_acc else 0.0
        ),
        "gtpo/sigma_ratio_of_means": (
            mean_sigma_g / mean_sigma_op if mean_sigma_op > 0 else 0.0
        ),
        "gtpo/gt_mean_contamination_per_group": (
            sum(contam_acc) / max(len(contam_acc), 1) if contam_acc else 0.0
        ),
        "gtpo/gt_mean_contamination_of_means": (
            abs(mean_mu_g - mean_mu_op) / mean_sigma_op if mean_sigma_op > 0 else 0.0
        ),
        "gtpo/degenerate_op_groups": float(degenerate_op_groups),
        "gtpo/sign_flip_count": float(sign_flip),
        "gtpo/sign_flip_rate": sign_flip / max(total_op, 1),
        "gtpo/groups_with_no_op_pos_baseline": float(no_op_pos_groups),
        "gtpo/n_groups": float(n_groups),
    }
    if gt_mags:
        out["gtpo/gt_adv_mag_mean"] = sum(gt_mags) / len(gt_mags)
        out["gtpo/gt_adv_mag_max"] = max(gt_mags)
    if op_pos_mags:
        out["gtpo/op_adv_pos_mean"] = sum(op_pos_mags) / len(op_pos_mags)
        out["gtpo/op_adv_pos_count"] = float(len(op_pos_mags))
    if op_neg_mags:
        out["gtpo/op_adv_neg_mean"] = sum(op_neg_mags) / len(op_neg_mags)
        out["gtpo/op_adv_neg_count"] = float(len(op_neg_mags))

    return out


# Per-row scalar metric keys to mirror from reward_metrics into
# ``batch.non_tensor_batch`` so downstream consumers (logging, diagnostics) can
# read them at the rollout level. ``iou_raw`` is the un-shaped IoU produced by
# TempSamp-R1 reward functions before GT_Soft transform; it is the signal
# strictly aligned with the eval metric.
_ROW_LEVEL_REWARD_KEYS_TO_PROPAGATE: tuple[str, ...] = ("iou_raw",)


def _propagate_per_row_reward_metrics(
    batch: DataProto,
    reward_metrics: dict[str, list[float]],
    keys: tuple[str, ...] = _ROW_LEVEL_REWARD_KEYS_TO_PROPAGATE,
) -> None:
    """Write selected per-row reward components into ``batch.non_tensor_batch``.

    ``reward_metrics`` is a dict-of-lists in row order (length == bsz). For
    keys present in ``reward_metrics`` we materialize an ndarray on the
    batch's non_tensor_batch so subsequent batch slicing keeps the per-row
    alignment. Idempotent (skips keys already present).
    """
    bsz = len(batch)
    for key in keys:
        if key in batch.non_tensor_batch:
            continue
        values = reward_metrics.get(key)
        if values is None:
            continue
        if len(values) != bsz:
            continue
        arr = np.asarray(values, dtype=np.float32)
        # Replace NaN/Inf with 0 so downstream torch ops never explode.
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        batch.non_tensor_batch[key] = arr


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        # Best-train ckpt tracking (separate from rolling save_limit window).
        # See TrainerConfig.keep_best_train_ckpt for rationale. The buffer is a
        # FIFO of recent metric values; we save when the smoothed value strictly
        # improves and we have already passed best_train_min_step.
        self._best_train_window: deque[float] = deque(
            maxlen=max(1, int(config.trainer.best_train_smooth_window))
        )
        self._best_train_score: float = float("-inf")
        self._best_train_step: int = 0

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # GTPO: optionally inject GT-built response into one rollout per group
        self._gt_builder = None
        if config.algorithm.use_gt_injection:
            if config.worker.rollout.n <= 1:
                raise ValueError("algorithm.use_gt_injection requires worker.rollout.n > 1.")
            if not config.algorithm.gt_builder:
                raise ValueError(
                    "algorithm.use_gt_injection=True requires algorithm.gt_builder to be set "
                    "(e.g. 'scripts.timelens.timelens_reward:build_gt_response')."
                )
            # GT injection rewrites one row in the group with GT-built tokens,
            # but the rollout_log_probs vLLM returned are for the original
            # online sample and therefore do not match those tokens. Combining
            # this with calculate_log_probs=True (bypass mode) would silently
            # corrupt the PPO ratio for the injected row, so the two options
            # are mutually exclusive — we force FSDP recomputation of
            # old_log_probs in that case.
            if getattr(config.worker.rollout, "calculate_log_probs", False):
                raise ValueError(
                    "algorithm.use_gt_injection=True is incompatible with "
                    "worker.rollout.calculate_log_probs=True: GT-injected rows would carry "
                    "vLLM logprobs from the original online sample (different tokens), "
                    "making PPO ratio meaningless. Set rollout.calculate_log_probs=False."
                )
            self._gt_builder = _load_dotted_callable(config.algorithm.gt_builder)
            replace_idx = config.algorithm.gt_replace_index
            if replace_idx != -1 and not (0 <= replace_idx < config.worker.rollout.n):
                raise ValueError(
                    f"algorithm.gt_replace_index must be -1 or in [0, {config.worker.rollout.n}), "
                    f"got {replace_idx}."
                )
            print(
                f"[GTPO] gt_injection ON | builder={config.algorithm.gt_builder} "
                f"| replace_index={replace_idx} (n={config.worker.rollout.n}) "
                f"| log_exclude={config.algorithm.gt_log_exclude}"
            )

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        rollout_rows = config.data.rollout_batch_size * config.worker.rollout.n
        actor_global_batch_size = config.worker.actor.global_batch_size
        self.skip_old_log_probs = (
            _skip_old_log_probs_enabled()
            and not self.use_reference_policy
            and config.worker.actor.ppo_epochs == 1
            and actor_global_batch_size in {config.data.rollout_batch_size, rollout_rows}
        )
        if _skip_old_log_probs_enabled() and not self.skip_old_log_probs:
            print(
                "[trainer] VERL_SKIP_OLD_LOGPROBS=1 requested but disabled for this run: "
                "requires KL/reference policy disabled, actor.ppo_epochs=1, and one actor global mini-batch per rollout step."
            )
        elif self.skip_old_log_probs:
            print(
                "[trainer] VERL_SKIP_OLD_LOGPROBS=1: skipping FSDP old_log_probs recompute; "
                "actor update will use its first forward log_probs.detach() as old_log_probs."
            )

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        # Only update best_global_step on steps where _validate() actually ran.
        # The gating condition must match the main loop exactly
        # (val_reward_fn is not None && val_freq > 0 && global_step % val_freq == 0).
        # Otherwise val_reward_score stays at its 0.0 initial value while
        # best_val_reward_score starts at -1.0, so the trivial 0.0 > -1.0 check
        # would lock the very first checkpoint as "best" forever, causing
        # remove_obsolete_ckpt to keep the oldest checkpoint and silently delete
        # newer ones (symptom: save_limit=2 leaves "oldest + newest" on disk
        # instead of the two most recent).
        validation_ran_this_step = (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
            and self.global_step % self.config.trainer.val_freq == 0
        )
        if validation_ran_this_step and self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

        # Keep optimizer/extra_state/dataloader for the most recent checkpoint
        # only; thin older checkpoints down to model weights (huggingface/ +
        # model_*.pt) to save disk.
        if (
            self.config.trainer.keep_optim_only_latest
            and not self.config.trainer.save_model_only
        ):
            thin_out_old_ckpts(
                self.config.trainer.save_checkpoint_path,
                keep_full_step=self.global_step,
            )

    def _maybe_save_best_train_checkpoint(self, metrics: dict) -> None:
        """Save model-only ckpt when smoothed train metric hits a new best.

        Independent of save_freq / save_limit:
          - Lives under {save_checkpoint_path}/best_train/global_step_{N}/actor/
          - Always model-only (no optim/dataloader): for downstream eval only.
          - Old best is removed atomically when a new best is found, so the
            best_train/ subdir holds at most ONE checkpoint.

        Decision rule:
          smoothed = mean of last `best_train_smooth_window` values of
                     metrics[best_train_metric_key]
          if step >= best_train_min_step AND smoothed > self._best_train_score:
              save and update tracker
        """
        cfg = self.config.trainer
        if not getattr(cfg, "keep_best_train_ckpt", False):
            return

        metric_val = metrics.get(cfg.best_train_metric_key)
        if metric_val is None:
            return
        try:
            metric_val = float(metric_val)
        except (TypeError, ValueError):
            return

        self._best_train_window.append(metric_val)

        if self.global_step < int(cfg.best_train_min_step):
            return
        if len(self._best_train_window) < self._best_train_window.maxlen:
            # Wait until the smoothing buffer is full, otherwise early steps
            # have an artificially small window and bias toward early peaks.
            return

        smoothed = sum(self._best_train_window) / len(self._best_train_window)
        if smoothed <= self._best_train_score:
            return

        # New best: save model-only into a side directory.
        prev_best_step = self._best_train_step
        prev_best_score = self._best_train_score
        self._best_train_score = smoothed
        self._best_train_step = self.global_step

        best_root = os.path.join(cfg.save_checkpoint_path, "best_train")
        os.makedirs(best_root, exist_ok=True)
        new_dir = os.path.join(best_root, f"global_step_{self.global_step}")
        new_actor = os.path.join(new_dir, "actor")
        # Save current step (model only). Force save_model_only=True regardless
        # of the global flag so this side ckpt stays small.
        self.actor_rollout_ref_wg.save_checkpoint(new_actor, save_model_only=True)
        if self.use_critic:
            new_critic = os.path.join(new_dir, "critic")
            self.critic_wg.save_checkpoint(new_critic, save_model_only=True)

        # Tracker file lets downstream eval scripts auto-discover the best step.
        tracker = {
            "best_train_step": self._best_train_step,
            "best_train_metric_key": cfg.best_train_metric_key,
            "best_train_smoothed_value": round(self._best_train_score, 6),
            "best_train_smooth_window": self._best_train_window.maxlen,
            "best_train_actor_path": os.path.abspath(new_actor),
            "previous_best_step": prev_best_step,
            "previous_best_smoothed_value": (
                round(prev_best_score, 6) if prev_best_score != float("-inf") else None
            ),
        }
        with open(os.path.join(best_root, "best_train_tracker.json"), "w") as f:
            json.dump(tracker, f, ensure_ascii=False, indent=2)

        # Evict the previous best ckpt directory (we only keep one in best_train/).
        if prev_best_step and prev_best_step != self.global_step:
            old_dir = os.path.join(best_root, f"global_step_{prev_best_step}")
            if os.path.isdir(old_dir):
                import shutil
                try:
                    shutil.rmtree(old_dir)
                except OSError as exc:
                    print(f"[best-train ckpt] WARN: failed to remove {old_dir}: {exc}")

        prev_score_str = (
            f"{prev_best_score:.4f}" if prev_best_score != float("-inf") else "n/a"
        )
        print(
            f"[best-train ckpt] step={self.global_step} "
            f"smoothed_{cfg.best_train_metric_key}={self._best_train_score:.4f} "
            f"(prev best step={prev_best_step or 0}, value={prev_score_str}) "
            f"→ {new_actor}",
            flush=True,
        )

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _assert_multimodal_contract(self, data: DataProto, stage: str) -> None:
        if "multi_modal_data" not in data.non_tensor_batch:
            return

        problem_ids = data.non_tensor_batch.get("problem_id", None)
        uids = data.non_tensor_batch.get("uid", None)
        for idx, multi_modal_data in enumerate(data.non_tensor_batch["multi_modal_data"]):
            try:
                validate_multi_modal_data_contract(multi_modal_data)
            except Exception as exc:
                problem_id = None if problem_ids is None else problem_ids[idx]
                uid = None if uids is None else uids[idx]
                raise ValueError(
                    f"{stage}: invalid multi_modal_data at index={idx}, uid={uid}, problem_id={problem_id}: {exc}"
                ) from exc

    def _maybe_log_val_generations(
        self,
        inputs: list[str],
        outputs: list[str],
        labels: list[str],
        scores: list[float],
        problem_ids: list[Any],
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, label, score, problem_id) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores, problem_ids))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids = [], [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.val_video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.val_video_max_pixels
            test_gen_batch.meta_info["video_total_pixels"] = self.config.data.val_video_total_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.val_video_fps
            test_gen_batch.meta_info["video_max_frames"] = self.config.data.val_video_max_frames

            self._assert_multimodal_contract(test_gen_batch, stage="validate")
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # Only pass fields needed by reward, excluding large multi_modal_data to reduce serialization
            val_reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(val_reward_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            if "problem_id" in test_batch.non_tensor_batch:
                sample_problem_ids.extend(test_batch.non_tensor_batch["problem_id"].tolist())
            else:
                sample_problem_ids.extend([None] * len(scores))

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(
            sample_inputs, sample_outputs, sample_labels, sample_scores, sample_problem_ids
        )
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            print("Sample problem_id:", sample_problem_ids[0])
            print("Sample prompt (with template):", sample_inputs[0])
            print("Sample response:", sample_outputs[0])
            print("Sample ground_truth:", sample_labels[0])
            print("Sample reward:", sample_scores[0])
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _inject_gt_rollout_in_gen_output(
        self,
        gen_batch_output: DataProto,
        ground_truths: Optional[np.ndarray],
        extras: Optional[np.ndarray],
        n: int,
    ) -> int:
        """GTPO: overwrite one rollout per group with a GT-built response.

        The replacement text comes from `self._gt_builder(ground_truth, extra)`.
        The replaced index within the group is `algorithm.gt_replace_index`
        (-1 -> last).

        Also writes a bool array `gen_batch_output.non_tensor_batch["is_gt_row"]`
        flagging which rows are GT, so later stages (logging, optional filtering)
        can tell them apart from on-policy samples.
        """
        if not self.config.algorithm.use_gt_injection or self._gt_builder is None:
            return 0
        if ground_truths is None or len(ground_truths) == 0:
            return 0

        replace_index = self.config.algorithm.gt_replace_index
        if replace_index == -1:
            replace_index = n - 1

        responses = gen_batch_output.batch["responses"].clone()
        response_length = responses.size(1)
        device = responses.device
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = gen_batch_output.meta_info.get("eos_token_id", self.tokenizer.eos_token_id)
        # `get_response_mask` supports list[int]; single-eos tail append needs a scalar.
        eos_single = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id

        is_gt_row = np.zeros(responses.size(0), dtype=bool)
        replaced = 0
        for i, gt_raw in enumerate(ground_truths):
            if gt_raw is None:
                continue
            extra = extras[i] if extras is not None and i < len(extras) else {}
            if extra is None:
                extra = {}
            try:
                gt_text = self._gt_builder(str(gt_raw), extra)
            except Exception as err:
                print(f"[GTPO] gt_builder failed at idx={i} gt={gt_raw!r}: {err}")
                continue
            if not isinstance(gt_text, str) or len(gt_text.strip()) == 0:
                continue

            gt_token_ids = self.tokenizer.encode(gt_text, add_special_tokens=False)
            # Ensure the GT response terminates with EOS so response_mask covers it.
            if eos_single is not None and (len(gt_token_ids) == 0 or gt_token_ids[-1] != eos_single):
                gt_token_ids = gt_token_ids + [eos_single]
            if len(gt_token_ids) > response_length:
                # Keep the tail so EOS survives; pad head is unusual, just truncate front.
                gt_token_ids = gt_token_ids[-response_length:]

            gt_tokens = VF.pad_2d_list_to_length(
                [gt_token_ids],
                pad_token_id,
                max_length=response_length,
            ).to(device)

            replace_idx = i * n + replace_index
            responses[replace_idx] = gt_tokens.squeeze(0)
            is_gt_row[replace_idx] = True
            replaced += 1

        if replaced == 0:
            return 0

        prompts = gen_batch_output.batch["prompts"]
        prompt_length = prompts.size(-1)
        attention_mask = gen_batch_output.batch["attention_mask"]
        position_ids = gen_batch_output.batch["position_ids"]

        prompt_attention_mask = attention_mask[..., :prompt_length]
        prompt_position_ids = position_ids[..., :prompt_length]

        response_mask = VF.get_response_mask(responses, eos_token_id=eos_token_id, dtype=prompt_attention_mask.dtype)
        sequence_ids = torch.cat([prompts, responses], dim=-1)

        batch_size = responses.size(0)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if prompt_position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, prompt_position_ids.size(1), -1
            )

        response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
        full_attention_mask = torch.cat((prompt_attention_mask, response_mask), dim=-1)

        gen_batch_output.batch["responses"] = responses
        gen_batch_output.batch["input_ids"] = sequence_ids
        gen_batch_output.batch["response_mask"] = response_mask
        gen_batch_output.batch["position_ids"] = full_position_ids
        gen_batch_output.batch["attention_mask"] = full_attention_mask
        gen_batch_output.non_tensor_batch["is_gt_row"] = is_gt_row
        return replaced

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        print("Start generating batch...")
        try:
            batch_dict = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.train_dataloader)
            batch_dict = next(self.data_iterator)

        meta_info = {
            "image_min_pixels": self.config.data.image_min_pixels,
            "image_max_pixels": self.config.data.image_max_pixels,
            "video_min_pixels": self.config.data.video_min_pixels,
            "video_max_pixels": self.config.data.video_max_pixels,
            "video_total_pixels": self.config.data.video_total_pixels,
            "video_fps": self.config.data.video_fps,
            "video_max_frames": self.config.data.video_max_frames,
        }
        new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )

        gen_batch = new_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            meta_info_keys=[
                "image_min_pixels",
                "image_max_pixels",
                "video_min_pixels",
                "video_max_pixels",
                "video_total_pixels",
                "video_fps",
                "video_max_frames",
            ],
        )

        self._assert_multimodal_contract(gen_batch, stage="train")

        gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

        if self.config.algorithm.adv_estimator == "remax":
            gen_baseline_batch = deepcopy(gen_batch)
            gen_baseline_batch.meta_info["temperature"] = 0
            gen_baseline_batch.meta_info["n"] = 1
            gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

            new_batch = new_batch.union(gen_baseline_output)
            remax_reward_batch = new_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in new_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(remax_reward_batch))
            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
            new_batch.batch["reward_baselines"] = reward_baseline_tensor
            del gen_baseline_batch, gen_baseline_output

        # GTPO: overwrite one rollout per group with a GT-built response (oracle anchor).
        if self.config.algorithm.use_gt_injection:
            gt_values = new_batch.non_tensor_batch.get("ground_truth", None)
            if gt_values is None:
                raise KeyError(
                    "algorithm.use_gt_injection=True but `ground_truth` is missing from the "
                    "batch non_tensor_batch. Check your dataset's answer_key."
                )
            gt_injected = self._inject_gt_rollout_in_gen_output(
                gen_batch_output=gen_batch_output,
                ground_truths=gt_values,
                extras=None,
                n=self.config.worker.rollout.n,
            )
            metrics["gtpo/groups_with_gt_injected"] = int(gt_injected)

        new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_batch_output)

        return new_batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = _NoOpProgress() if _disable_tqdm() else tqdm(
            range(self.training_steps),
            desc="Running step",
            position=0,
        )
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        train_start_time = time.time()
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward asynchronously so it can overlap with old-log-prob compute.
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_batch = batch.select(
                            batch_keys=["responses", "response_mask"],
                            non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                        )
                        reward_ref = self.reward_fn.compute_reward.remote(reward_batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    # Actor update always needs the rollout temperature to compute response log-probs.
                    batch.meta_info["temperature"] = self.config.worker.rollout.temperature
                    if "rollout_log_probs" in batch.batch:
                        # Bypass mode: vLLM already returned a logprob for every
                        # sampled token during rollout, so we reuse them as
                        # old_log_probs. This skips one FSDP recomputation, and
                        # the first PPO mini-batch update sees ratio != 1, which
                        # lets PPO clip act as a real trust region.
                        # vLLM logprobs are computed as log_softmax(logits /
                        # temperature), exactly matching actor.compute_log_prob,
                        # so the values are directly comparable.
                        rollout_lp = batch.batch.pop("rollout_log_probs")
                        batch.batch["old_log_probs"] = rollout_lp.to(torch.float32)
                    elif self.skip_old_log_probs:
                        batch.meta_info["skip_old_log_probs"] = True
                        metrics["actor/old_log_probs_skipped"] = 1.0
                    else:
                        old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                        batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        _propagate_per_row_reward_metrics(batch, reward_metrics)
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                    )
                    # GTPO diagnostics: emit per-step μ_op/μ_g/σ_op/σ_g and
                    # GT-row contamination stats. No-op for non-GTPO runs.
                    gtpo_diag = compute_gtpo_advantage_diagnostics(batch)
                    if gtpo_diag:
                        metrics.update(gtpo_diag)

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            # Best-train ckpt: track every step (decoupled from save_freq) and
            # save model-only into a side dir when smoothed reward improves.
            with timer("save_best_train_checkpoint", timing_raw):
                self._maybe_save_best_train_checkpoint(metrics)

            if _print_step_summary_enabled():
                step_time = metrics.get("timing_s/step")
                elapsed = time.time() - train_start_time
                avg_step_time = elapsed / max(1, self.global_step)
                remaining_steps = max(0, self.training_steps - self.global_step)
                eta = avg_step_time * remaining_steps
                print(
                    "[TRAIN STEP] "
                    f"{self.global_step}/{self.training_steps} "
                    f"step_s={_fmt_metric(step_time, precision=1)} "
                    f"elapsed={_fmt_duration(elapsed)} "
                    f"eta={_fmt_duration(eta)}",
                    flush=True,
                )

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training (skip entirely when val_freq <= 0)
        if self.val_reward_fn is not None and self.config.trainer.val_freq > 0:
            if (
                val_metrics is None
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
