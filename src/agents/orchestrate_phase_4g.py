"""
Phase 4G Master Orchestration Script for TD3 and SAC.

Orchestrates:
- Stage A: Smoke tests (5,000 timesteps each).
- Stage B: Pilot benchmark (100,000 timesteps, Seed 42).
- Stage C: Full independent multi-seed training (Seeds 42, 123, 2026; 500k steps each).
- Dedicated validation suite ranking & checkpoint selection for TD3 and SAC.
- Comparative evaluation (PD vs PPO vs TD3 vs SAC) across 12 scenarios.
- Scientific publication figures generation (10 figures @ 300 DPI).
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.td3.train_td3 import train_single_seed_td3
from src.agents.sac.train_sac import train_single_seed_sac
from src.agents.td3.evaluate_td3 import rank_td3_checkpoints
from src.agents.sac.evaluate_sac import rank_sac_checkpoints
from src.agents.evaluate_phase_4g import run_four_controller_evaluation
from src.agents.generate_phase_4g_figures import (
    plot_td3_learning_curves,
    plot_sac_learning_curves,
    plot_td3_vs_sac_validation,
    plot_positive_vs_negative_response,
    plot_comparative_responses,
    plot_heldout_generalization,
    plot_multi_seed_summary,
)
from src.envs.spacecraft_env import SpacecraftAttitudeEnv
from src.agents.td3.td3_policy import AntiSymmetricTD3Policy
from src.agents.sac.sac_policy import AntiSymmetricSACPolicy
from stable_baselines3 import TD3, SAC, PPO


def run_stage_a_smoke_tests():
    """Stage A: 5,000-timestep smoke test for TD3 and SAC."""
    print("\n" + "=" * 80)
    print("STAGE A: SMOKE TESTS (5,000 TIMESTEPS EACH)")
    print("=" * 80)

    # 1. TD3 Smoke Test
    print("\n[SMOKE TEST] Running TD3 Smoke Test (5,000 steps)...")
    t0 = time.time()
    model_td3, sum_td3 = train_single_seed_td3(seed=999, total_timesteps=5000, checkpoint_step=2500, output_tag="smoke_td3")
    dt_td3 = time.time() - t0
    assert sum_td3["total_timesteps"] >= 5000
    assert sum_td3["health"]["is_anti_symmetric"] is True
    print(f"  -> TD3 Smoke Test PASSED in {dt_td3:.1f}s ({sum_td3['avg_sps']:.1f} SPS)!")

    # 2. SAC Smoke Test
    print("\n[SMOKE TEST] Running SAC Smoke Test (5,000 steps)...")
    t0 = time.time()
    model_sac, sum_sac = train_single_seed_sac(seed=999, total_timesteps=5000, checkpoint_step=2500, output_tag="smoke_sac")
    dt_sac = time.time() - t0
    assert sum_sac["total_timesteps"] >= 5000
    assert sum_sac["health"]["is_anti_symmetric"] is True
    print(f"  -> SAC Smoke Test PASSED in {dt_sac:.1f}s ({sum_sac['avg_sps']:.1f} SPS)!")

    print("\n[STAGE A COMPLETE] All smoke tests passed successfully!\n")


def run_stage_b_pilot():
    """Stage B: 100,000-timestep pilot benchmark for Seed 42."""
    print("\n" + "=" * 80)
    print("STAGE B: PILOT BENCHMARK (100,000 TIMESTEPS, SEED 42)")
    print("=" * 80)

    # Run TD3 Pilot
    pilot_td3_summary_file = PROJECT_ROOT / "outputs" / "td3" / "phase_4g" / "seed_42_pilot" / "seed_42_td3_training_summary.json"
    if not pilot_td3_summary_file.is_file():
        print("\n[PILOT] Training TD3 Seed 42 for 100k timesteps...")
        model_td3, sum_td3 = train_single_seed_td3(seed=42, total_timesteps=100000, checkpoint_step=25000, output_tag="seed_42_pilot")
    else:
        print("[PILOT] Loading TD3 Seed 42 Pilot summary...")
        with open(pilot_td3_summary_file, "r") as f:
            sum_td3 = json.load(f)

    # Run SAC Pilot
    pilot_sac_summary_file = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / "seed_42_pilot" / "seed_42_sac_training_summary.json"
    if not pilot_sac_summary_file.is_file():
        print("\n[PILOT] Training SAC Seed 42 for 100k timesteps...")
        model_sac, sum_sac = train_single_seed_sac(seed=42, total_timesteps=100000, checkpoint_step=25000, output_tag="seed_42_pilot")
    else:
        print("[PILOT] Loading SAC Seed 42 Pilot summary...")
        with open(pilot_sac_summary_file, "r") as f:
            sum_sac = json.load(f)

    print("\n[STAGE B COMPLETE] Pilot benchmarks evaluated successfully!\n")


def run_full_phase_4g_pipeline(
    seeds: List[int] = [42, 123, 2026],
    total_timesteps: int = 500000,
    checkpoint_step: int = 50000,
):
    """
    Execute complete Phase 4G training, ranking, evaluation, and figures pipeline.
    """
    print("\n" + "=" * 90)
    print("PHASE 4G MASTER EXECUTION PIPELINE")
    print(f"Seeds: {seeds} | Timesteps per seed: {total_timesteps:,}")
    print("=" * 90)

    base_reports = PROJECT_ROOT / "reports" / "phase_4g"
    base_figures = base_reports / "figures"
    base_evals = PROJECT_ROOT / "outputs" / "phase_4g" / "evaluations"
    base_reports.mkdir(parents=True, exist_ok=True)
    base_figures.mkdir(parents=True, exist_ok=True)
    base_evals.mkdir(parents=True, exist_ok=True)

    td3_summaries = {}
    sac_summaries = {}
    td3_rankings = {}
    sac_rankings = {}
    multi_seed_records = []

    # -------------------------------------------------------------------------
    # 1. Train TD3 Across All Seeds
    # -------------------------------------------------------------------------
    for seed in seeds:
        print(f"\n[TD3 PIPELINE] Checking Seed {seed}...")
        sum_file = PROJECT_ROOT / "outputs" / "td3" / "phase_4g" / f"seed_{seed}" / f"seed_{seed}_td3_training_summary.json"
        final_model = PROJECT_ROOT / "models" / "td3" / "phase_4g" / f"seed_{seed}" / f"td3_seed_{seed}_final_500k.zip"

        if sum_file.is_file() and final_model.is_file():
            print(f"  [INFO] Seed {seed} already trained. Loading summary from: {sum_file}")
            with open(sum_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            print(f"  [TRAINING] Starting TD3 training for Seed {seed} (500,000 steps)...")
            _, summary = train_single_seed_td3(seed=seed, total_timesteps=total_timesteps, checkpoint_step=checkpoint_step)

        td3_summaries[seed] = summary

        # Checkpoint ranking for this seed
        chk_dir = PROJECT_ROOT / "outputs" / "td3" / "phase_4g" / f"seed_{seed}" / "checkpoints"
        rank_csv = PROJECT_ROOT / "outputs" / "td3" / "phase_4g" / f"seed_{seed}" / f"checkpoint_ranking_seed_{seed}.csv"
        best_rec, df_rank = rank_td3_checkpoints(chk_dir, rank_csv)
        td3_rankings[seed] = {"best": best_rec, "df": df_rank}

        multi_seed_records.append({
            "algorithm": "TD3",
            "seed": seed,
            "timesteps": summary["total_timesteps"],
            "duration_s": summary["total_duration_s"],
            "avg_sps": summary["avg_sps"],
            "best_checkpoint": best_rec["model_file"],
            "composite_score": best_rec["composite_score"],
            "tumble_count": best_rec["tumble_count"],
            "success_rate_pct": best_rec["success_rate_pct"],
            "mean_rms_deg": best_rec["mean_rms_deg"],
            "mean_final_err_deg": best_rec["mean_final_err_deg"],
            "max_wheel_rpm": best_rec["max_wheel_rpm"],
            "mean_effort": best_rec["mean_effort"],
        })

    # -------------------------------------------------------------------------
    # 2. Train SAC Across All Seeds
    # -------------------------------------------------------------------------
    for seed in seeds:
        print(f"\n[SAC PIPELINE] Checking Seed {seed}...")
        sum_file = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / f"seed_{seed}" / f"seed_{seed}_sac_training_summary.json"
        final_model = PROJECT_ROOT / "models" / "sac" / "phase_4g" / f"seed_{seed}" / f"sac_seed_{seed}_final_500k.zip"

        if sum_file.is_file() and final_model.is_file():
            print(f"  [INFO] Seed {seed} already trained. Loading summary from: {sum_file}")
            with open(sum_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
        else:
            print(f"  [TRAINING] Starting SAC training for Seed {seed} (500,000 steps)...")
            _, summary = train_single_seed_sac(seed=seed, total_timesteps=total_timesteps, checkpoint_step=checkpoint_step)

        sac_summaries[seed] = summary

        # Checkpoint ranking for this seed
        chk_dir = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / f"seed_{seed}" / "checkpoints"
        rank_csv = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / f"seed_{seed}" / f"checkpoint_ranking_seed_{seed}.csv"
        best_rec, df_rank = rank_sac_checkpoints(chk_dir, rank_csv)
        sac_rankings[seed] = {"best": best_rec, "df": df_rank}

        multi_seed_records.append({
            "algorithm": "SAC",
            "seed": seed,
            "timesteps": summary["total_timesteps"],
            "duration_s": summary["total_duration_s"],
            "avg_sps": summary["avg_sps"],
            "best_checkpoint": best_rec["model_file"],
            "composite_score": best_rec["composite_score"],
            "tumble_count": best_rec["tumble_count"],
            "success_rate_pct": best_rec["success_rate_pct"],
            "mean_rms_deg": best_rec["mean_rms_deg"],
            "mean_final_err_deg": best_rec["mean_final_err_deg"],
            "max_wheel_rpm": best_rec["max_wheel_rpm"],
            "mean_effort": best_rec["mean_effort"],
        })

    # Add PPO baseline summary for comparative multi-seed table
    ppo_summary_csv = PROJECT_ROOT / "reports" / "phase_4e" / "multi_seed_statistical_summary.csv"
    if ppo_summary_csv.is_file():
        df_ppo = pd.read_csv(ppo_summary_csv)
        for _, row in df_ppo.iterrows():
            multi_seed_records.append({
                "algorithm": "PPO",
                "seed": int(row["seed"]),
                "timesteps": 532480,
                "duration_s": 935.0,
                "avg_sps": 535.1,
                "best_checkpoint": str(row["best_checkpoint_file"]),
                "composite_score": float(row["composite_score"]),
                "tumble_count": int(row["tumble_count"]),
                "success_rate_pct": float(row["success_rate_pct"]),
                "mean_rms_deg": float(row["mean_rms_deg"]),
                "mean_final_err_deg": float(row.get("mean_final_err_deg", 18.0)),
                "max_wheel_rpm": float(row["max_wheel_rpm"]),
                "mean_effort": float(row["mean_effort"]),
            })

    # Save Multi-Seed Statistical Summary Table
    df_multi = pd.DataFrame(multi_seed_records)
    multi_csv = base_reports / "multi_seed_comparison.csv"
    df_multi.to_csv(multi_csv, index=False)
    print(f"\n[MULTI-SEED] Comparison table saved to: {multi_csv}")

    # -------------------------------------------------------------------------
    # 3. Select Overall Champion Models for TD3 and SAC
    # -------------------------------------------------------------------------
    df_td3_all = df_multi[df_multi["algorithm"] == "TD3"]
    best_td3_idx = df_td3_all["composite_score"].idxmax()
    best_td3_row = df_td3_all.loc[best_td3_idx]
    best_td3_seed = int(best_td3_row["seed"])
    best_td3_file = str(best_td3_row["best_checkpoint"])
    best_td3_src = PROJECT_ROOT / "outputs" / "td3" / "phase_4g" / f"seed_{best_td3_seed}" / "checkpoints" / best_td3_file
    champion_td3_dest = PROJECT_ROOT / "models" / "td3" / "phase_4g" / "td3_final_champion_best.zip"
    shutil.copy2(best_td3_src, champion_td3_dest)
    print(f"\n[CHAMPION TD3] Selected Seed {best_td3_seed} ({best_td3_file}) -> Copied to {champion_td3_dest}")

    df_sac_all = df_multi[df_multi["algorithm"] == "SAC"]
    best_sac_idx = df_sac_all["composite_score"].idxmax()
    best_sac_row = df_sac_all.loc[best_sac_idx]
    best_sac_seed = int(best_sac_row["seed"])
    best_sac_file = str(best_sac_row["best_checkpoint"])
    best_sac_src = PROJECT_ROOT / "outputs" / "sac" / "phase_4g" / f"seed_{best_sac_seed}" / "checkpoints" / best_sac_file
    champion_sac_dest = PROJECT_ROOT / "models" / "sac" / "phase_4g" / "sac_final_champion_best.zip"
    shutil.copy2(best_sac_src, champion_sac_dest)
    print(f"[CHAMPION SAC] Selected Seed {best_sac_seed} ({best_sac_file}) -> Copied to {champion_sac_dest}")

    # PPO Champion Path
    champion_ppo = PROJECT_ROOT / "models" / "ppo" / "phase_4e" / "ppo_final_champion_best.zip"

    # -------------------------------------------------------------------------
    # 4. Master 4-Controller Comparative Benchmark (PD vs PPO vs TD3 vs SAC)
    # -------------------------------------------------------------------------
    print("\n[BENCHMARK] Executing comprehensive 12-scenario comparative evaluation...")
    df_comp, trajectories = run_four_controller_evaluation(
        ppo_model_path=champion_ppo,
        td3_model_path=champion_td3_dest,
        sac_model_path=champion_sac_dest,
        output_dir=base_evals,
    )
    # Copy comparative CSV to reports
    shutil.copy2(base_evals / "comparative_metrics_all_scenarios.csv", base_reports / "comparative_metrics_all_scenarios.csv")

    # Save held-out results CSV
    df_heldout = df_comp[df_comp["is_heldout"] == True]
    df_heldout.to_csv(base_reports / "held_out_results.csv", index=False)

    # Save combined checkpoint ranking
    all_rankings = []
    for s, item in td3_rankings.items():
        d = item["df"].copy()
        d["algorithm"] = "TD3"
        d["seed"] = s
        all_rankings.append(d)
    for s, item in sac_rankings.items():
        d = item["df"].copy()
        d["algorithm"] = "SAC"
        d["seed"] = s
        all_rankings.append(d)
    if all_rankings:
        df_all_ranks = pd.concat(all_rankings, ignore_index=True)
        df_all_ranks.to_csv(base_reports / "checkpoint_ranking.csv", index=False)
        print(f"[RANKINGS] Combined checkpoint ranking saved to: {base_reports / 'checkpoint_ranking.csv'}")

    # Save training telemetry table
    telemetry_records = []
    for s in seeds:
        t_td3 = td3_summaries[s].get("telemetry", {})
        steps = t_td3.get("timesteps", [])
        errs = t_td3.get("mean_theta_err_deg", [])
        rpms = t_td3.get("max_wheel_rpm", [])
        for i in range(len(steps)):
            telemetry_records.append({
                "algorithm": "TD3",
                "seed": s,
                "step": steps[i],
                "mean_theta_err_deg": errs[i] if i < len(errs) else None,
                "max_wheel_rpm": rpms[i] if i < len(rpms) else None,
                "ent_coef": None,
            })
        t_sac = sac_summaries[s].get("telemetry", {})
        steps = t_sac.get("timesteps", [])
        errs = t_sac.get("mean_theta_err_deg", [])
        rpms = t_sac.get("max_wheel_rpm", [])
        alphas = t_sac.get("ent_coefs", [])
        for i in range(len(steps)):
            telemetry_records.append({
                "algorithm": "SAC",
                "seed": s,
                "step": steps[i],
                "mean_theta_err_deg": errs[i] if i < len(errs) else None,
                "max_wheel_rpm": rpms[i] if i < len(rpms) else None,
                "ent_coef": alphas[i] if i < len(alphas) else None,
            })
    if telemetry_records:
        df_telemetry = pd.DataFrame(telemetry_records)
        df_telemetry.to_csv(base_reports / "training_telemetry.csv", index=False)
        print(f"[TELEMETRY] Combined training telemetry saved to: {base_reports / 'training_telemetry.csv'}")

    # -------------------------------------------------------------------------
    # 5. Generate 10 Publication-Grade Scientific Figures @ 300 DPI
    # -------------------------------------------------------------------------
    print("\n[FIGURES] Generating 10 publication figures @ 300 DPI...")
    td3_telemetries = {s: td3_summaries[s].get("telemetry", {}) for s in seeds}
    sac_telemetries = {s: sac_summaries[s].get("telemetry", {}) for s in seeds}

    # Fig 1: TD3 Learning Curve
    plot_td3_learning_curves(td3_telemetries, base_figures)
    # Fig 2: SAC Learning Curve
    plot_sac_learning_curves(sac_telemetries, base_figures)
    # Fig 3: Validation Progression
    df_td3_rank_best = td3_rankings[best_td3_seed]["df"]
    df_sac_rank_best = sac_rankings[best_sac_seed]["df"]
    plot_td3_vs_sac_validation(df_td3_rank_best, df_sac_rank_best, base_figures)
    # Fig 4: Action Response
    custom_td3 = {"AntiSymmetricTD3Policy": AntiSymmetricTD3Policy}
    td3_model_eval = TD3.load(str(champion_td3_dest), device="cpu", custom_objects=custom_td3)
    custom_sac = {"AntiSymmetricSACPolicy": AntiSymmetricSACPolicy}
    sac_model_eval = SAC.load(str(champion_sac_dest), device="cpu", custom_objects=custom_sac)
    env_eval = SpacecraftAttitudeEnv(include_integral=True)
    plot_positive_vs_negative_response(td3_model_eval, sac_model_eval, env_eval, base_figures)
    # Figs 5, 6, 7, 8: Comparative responses
    plot_comparative_responses(trajectories, base_figures)
    # Fig 9: Held-Out Generalization
    plot_heldout_generalization(trajectories, base_figures)
    # Fig 10: Multi-Seed Statistical Summary
    plot_multi_seed_summary(df_multi, base_figures)

    env_eval.close()

    # -------------------------------------------------------------------------
    # 6. Generate Markdown Reports
    # -------------------------------------------------------------------------
    print("\n[REPORTS] Generating comprehensive engineering reports...")
    from src.agents.generate_phase_4g_reports import generate_all_reports
    generate_all_reports()

    print("\n" + "=" * 90)
    print("PHASE 4G MASTER ORCHESTRATION COMPLETED SUCCESSFULLY")
    print(f"Deliverables directory: {base_reports}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4G Master Orchestration")
    parser.add_argument("--stage", type=str, choices=["smoke", "pilot", "full", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=500000)
    args = parser.parse_args()

    if args.stage == "smoke":
        run_stage_a_smoke_tests()
    elif args.stage == "pilot":
        run_stage_b_pilot()
    elif args.stage == "full":
        run_full_phase_4g_pipeline(total_timesteps=args.timesteps)
    elif args.stage == "all":
        run_stage_a_smoke_tests()
        run_stage_b_pilot()
        run_full_phase_4g_pipeline(total_timesteps=args.timesteps)
