from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from hilbert_rownorm.config import ModelConfig
from hilbert_rownorm.run_metadata import build_run_config
from hilbert_rownorm.run_plan import RunPlan


class FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [
            {
                "role": "backbone",
                "base_lr": 0.01,
                "lr": 0.005,
                "weight_decay": 0.1,
                "betas": (0.8, 0.9),
                "eps": 1e-8,
            }
        ]


def test_run_config_records_the_retained_geometry_controls(tmp_path: Path) -> None:
    args = Namespace(
        model="tiny",
        optimizer="adamw",
        lm_head_optimizer="adamw",
        exact_head_diameter_every_log=True,
        exact_head_diameter_block_rows=64,
        hilbert_step_probe_tokens=128,
        data=tmp_path / "data",
        output=tmp_path / "run",
        micro_batch=2,
        global_batch=8,
        tokens_per_parameter=3,
        seed=17,
        max_steps=5,
        warmup_fraction=0.1,
        min_lr_ratio=0.2,
        max_gradient_norm=1.5,
        no_gradient_clipping=True,
        log_every=4,
        eval_every=5,
        eval_tokens=100,
        save_every=6,
        gradient_checkpointing=True,
        require_cuda=False,
        wandb_project="project",
        wandb_entity="entity",
        wandb_run_name="named-run",
        wandb_mode="disabled",
    )
    config = ModelConfig(
        dim=4,
        n_layers=1,
        head_dim=2,
        vocab_size=8,
        max_seq_len=16,
    )
    plan = RunPlan(
        accumulation_steps=2,
        tokens_per_step=128,
        target_tokens=999,
        schedule_steps=7,
        run_steps=5,
        warmup_steps=1,
    )
    optimizer_config = {"groups": {"backbone": {"lr": 0.01}}}

    result = build_run_config(
        args,
        config,
        "adamw-backbone:tiny:adamw:adamw",
        optimizer_config,
        [FakeOptimizer()],  # type: ignore[list-item]
        SimpleNamespace(shards=("train-0", "train-1")),
        SimpleNamespace(shards=("val-0",)),
        world_size=2,
        plan=plan,
        actual_val_tokens=96,
        device_name="CPU",
    )

    assert result == {
        "model": "tiny",
        "optimizer": "adamw",
        "lm_head_optimizer": "adamw",
        "exact_head_diameter_every_log": True,
        "exact_head_diameter_block_rows": 64,
        "hilbert_step_probe_tokens": 128,
        "dataset": "FineWeb-100B (GPT-2 tokenizer)",
        "micro_batch": 2,
        "global_batch": 8,
        "tokens_per_parameter": 3,
        "seed": 17,
        "max_steps": 5,
        "warmup_fraction": 0.1,
        "min_lr_ratio": 0.2,
        "max_gradient_norm": 1.5,
        "no_gradient_clipping": True,
        "log_every": 4,
        "eval_every": 5,
        "eval_tokens": 100,
        "save_every": 6,
        "gradient_checkpointing": True,
        "require_cuda": False,
        "optimizer_recipe_id": "adamw-backbone:tiny:adamw:adamw",
        "optimizer_config_values": optimizer_config,
        "model_config": {
            "dim": 4,
            "n_layers": 1,
            "head_dim": 2,
            "mlp_expansion": 4,
            "vocab_size": 8,
            "max_seq_len": 16,
            "rope_theta": 10_000.0,
            "rms_norm_eps": 1e-6,
            "initializer_std": 0.02,
        },
        "parameter_count": config.parameter_count,
        "optimizer_groups": [
            {
                "optimizer": "FakeOptimizer",
                "role": "backbone",
                "peak_lr": 0.01,
                "weight_decay": 0.1,
                "betas": [0.8, 0.9],
                "eps": 1e-8,
            }
        ],
        "train_shards": 2,
        "validation_shards": 1,
        "world_size": 2,
        "gradient_accumulation_steps": 2,
        "tokens_per_update": 128,
        "target_tokens": 999,
        "scheduled_tokens": 896,
        "schedule_steps": 7,
        "run_steps": 5,
        "schedule": "linear_warmup_cosine_decay",
        "warmup_steps": 1,
        "gradient_clipping": False,
        "actual_eval_tokens": 96,
        "precision": "FP32 params / BF16 compute / FP32 logits",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": "CPU",
    }
    obsolete = {
        "head_diagnostics_config",
        "head_diagnostics_config_values",
        "logit_diagnostics",
        "raw_head_gradient_snapshots",
        "exact_head_diameter_probes",
    }
    assert obsolete.isdisjoint(result)
    assert str(tmp_path) not in repr(result)
    assert "wandb_entity" not in result
