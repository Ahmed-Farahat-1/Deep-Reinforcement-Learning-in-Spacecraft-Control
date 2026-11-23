"""
Phase 4E Master Orchestration Script.

Orchestrates:
1. Multi-seed training (Seeds 42, 123, 2026 for 500k steps each).
2. Dedicated validation suite evaluation and checkpoint selection per seed.
3. Multi-seed statistical comparison and overall best model selection.
4. Full comparative evaluation of Best PPO vs Classical PD on 7 Benchmark + 5 Held-Out scenarios.
5. Generation of all 9 publication-grade figures @ 300 DPI.
"""

import sys
import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.train_phase_4e import train_single_seed
from src.agents.select_best_checkpoint import rank_and_select_best_checkpoint
from src.agents.evaluate_phase_4e import run_comparative_evaluation
from src.agents.generate_phase_4e_figures import (
    plot_learning_curves,
    plot_eval_reward_progression,
    plot_positive_vs_negative_response,
    plot_comparative_responses,
    plot_heldout_scenario_results,
    plot_multi_seed_comparison,
)
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.agents.anti_symmetric_policy import AntiSymmetricActorCriticPolicy
from stable_baselines3 import PPO


def run_phase_4e_workflow(
    seeds: List[int] = [42, 123, 2026],
    timesteps_per_seed: int = 500000,
    checkpoint_step: int = 100000,
):
    print("=" * 85)
    print("PHASE 4E: MASTER ORCHESTRATION PIPELINE")
    print(f"Seeds: {seeds} | Timesteps per Seed: {timesteps_per_seed:,}")
    print("=" * 85)

    base_reports = PROJECT_ROOT / "reports" / "phase_4e"
    base_figures = base_reports / "figures"
    base_evals = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4e" / "evaluations"
    base_models = PROJECT_ROOT / "models" / "ppo" / "phase_4e"

    for d in [base_reports, base_figures, base_evals, base_models]:
        d.mkdir(parents=True, exist_ok=True)

    seed_summaries = {}
    best_checkpoints_per_seed = {}
    telemetry_histories = {}
    eval_histories = {}
    multi_seed_metrics = {}

    # -------------------------------------------------------------------------
    # 1. Train Each Seed
    # -------------------------------------------------------------------------
    for seed in seeds:
        seed_final_model = PROJECT_ROOT / "models" / "ppo" / "phase_4e" / f"seed_{seed}" / f"ppo_seed_{seed}_final_500k.zip"
        seed_summary_file = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4e" / f"seed_{seed}" / f"seed_{seed}_training_summary.json"

        # Check if already trained to avoid redundant computation
        if seed_final_model.is_file() and seed_summary_file.is_file():
            print(f"\n[INFO] Seed {seed} already trained. Loading summary from: {seed_summary_file}")
            with open(seed_summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            print(f"\n[ORCHESTRATOR] Starting training for Seed {seed}...")
            _, summary = train_single_seed(
                seed=seed,
                total_timesteps=timesteps_per_seed,
                checkpoint_step=checkpoint_step,
            )

        seed_summaries[seed] = summary
        telemetry_histories[seed] = summary.get("telemetry", {})
        eval_histories[seed] = summary.get("eval_history", {})

        # ---------------------------------------------------------------------
        # 2. Dedicated Validation Suite Checkpoint Selection for this Seed
        # ---------------------------------------------------------------------
        chk_dir = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4e" / f"seed_{seed}" / "checkpoints"
        rank_csv = PROJECT_ROOT / "outputs" / "ppo_training" / "phase_4e" / f"seed_{seed}" / f"checkpoint_ranking_seed_{seed}.csv"
        best_rec, df_rank = rank_and_select_best_checkpoint(checkpoint_dir=chk_dir, output_csv=rank_csv)

        best_checkpoints_per_seed[seed] = best_rec
        multi_seed_metrics[seed] = {
            "success_rate_pct": float(best_rec["success_rate_pct"]),
            "mean_rms_deg": float(best_rec["mean_rms_deg"]),
            "mean_effort": float(best_rec["mean_effort"]),
            "composite_score": float(best_rec["composite_score"]),
            "max_wheel_rpm": float(best_rec["max_wheel_rpm"]),
            "tumble_count": int(best_rec["tumble_count"]),
            "best_checkpoint_file": str(best_rec["model_file"]),
            "best_checkpoint_path": str(best_rec["model_path"]),
        }

    # -------------------------------------------------------------------------
    # 3. Multi-Seed Statistical Summary Table
    # -------------------------------------------------------------------------
    df_multi = pd.DataFrame.from_dict(multi_seed_metrics, orient="index")
    df_multi.index.name = "seed"
    multi_csv = base_reports / "multi_seed_statistical_summary.csv"
    df_multi.to_csv(multi_csv)
    print(f"\n[ORCHESTRATOR] Multi-seed statistical summary saved to: {multi_csv}")

    # Select overall best model across all seeds
    best_overall_seed = int(df_multi["composite_score"].idxmax())
    best_overall_path = Path(multi_seed_metrics[best_overall_seed]["best_checkpoint_path"])
    print(f"\n[SELECTION] OVERALL CHAMPION MODEL: Seed {best_overall_seed} ({best_overall_path.name})")

    # Save to top-level models directory
    champion_dest = base_models / "ppo_final_champion_best.zip"
    shutil.copy2(best_overall_path, champion_dest)
    print(f"[SAVE] Champion model copied to: {champion_dest}")

    # -------------------------------------------------------------------------
    # 4. Comparative Evaluation: Champion PPO vs Classical PD (12 Scenarios)
    # -------------------------------------------------------------------------
    print("\n[ORCHESTRATOR] Running comprehensive comparative evaluation against Classical PD...")
    df_comp, trajectories = run_comparative_evaluation(
        ppo_model_path=champion_dest,
        output_dir=base_evals,
        include_heldout=True,
    )
    # Copy comparative CSV to reports
    shutil.copy2(base_evals / "comparative_metrics_ppo_vs_pd.csv", base_reports / "comparative_metrics_ppo_vs_pd.csv")

    # -------------------------------------------------------------------------
    # 5. Scientific Figures Generation (9 Figures @ 300 DPI)
    # -------------------------------------------------------------------------
    print("\n[ORCHESTRATOR] Generating 9 scientific publication figures @ 300 DPI...")
    custom_objects = {"AntiSymmetricActorCriticPolicy": AntiSymmetricActorCriticPolicy}
    champion_model = PPO.load(str(champion_dest), device="cpu", custom_objects=custom_objects)
    env_eval = SpacecraftAttitudeEnv(include_integral=True)

    # Fig 1: Training learning curves
    plot_learning_curves(telemetry_histories=telemetry_histories, output_dir=base_figures)
    # Fig 2: Evaluation reward progression
    plot_eval_reward_progression(eval_histories=eval_histories, output_dir=base_figures)
    # Fig 3: Anti-symmetric response
    plot_positive_vs_negative_response(model=champion_model, env=env_eval, output_dir=base_figures)
    # Figs 4, 5, 6, 7: Comparative attitude, rate, wheel, torque
    plot_comparative_responses(trajectories=trajectories, output_dir=base_figures)
    # Fig 8: Held-out generalization
    plot_heldout_scenario_results(trajectories=trajectories, output_dir=base_figures)
    # Fig 9: Multi-seed comparison
    plot_multi_seed_comparison(multi_seed_metrics=multi_seed_metrics, output_dir=base_figures)

    env_eval.close()
    print("\n" + "=" * 85)
    print("PHASE 4E MASTER ORCHESTRATION COMPLETED SUCCESSFULLY")
    print(f"Deliverables directory: {base_reports}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4E Master Workflow")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Seeds to train")
    parser.add_argument("--timesteps", type=int, default=500000, help="Timesteps per seed")
    parser.add_argument("--checkpoint-step", type=int, default=100000, help="Checkpoint interval")
    args = parser.parse_args()

    run_phase_4e_workflow(seeds=args.seeds, timesteps_per_seed=args.timesteps, checkpoint_step=args.checkpoint_step)
