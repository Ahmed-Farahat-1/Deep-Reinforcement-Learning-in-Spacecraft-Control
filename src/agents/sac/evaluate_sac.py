"""
SAC Evaluation & Checkpoint Selection Module.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC
from src.agents.sac.sac_policy import AntiSymmetricSACPolicy
from src.agents.select_best_checkpoint import (
    get_validation_suite,
    evaluate_checkpoint_on_validation_suite,
)


def rank_sac_checkpoints(
    checkpoint_dir: Path,
    output_csv: Optional[Path] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Scan all SAC checkpoint .zip files, evaluate on validation suite, and rank them."""
    if output_csv is not None and output_csv.is_file():
        df = pd.read_csv(output_csv)
        if len(df) > 0:
            df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)
            return df.iloc[0].to_dict(), df

    zip_files = sorted(list(checkpoint_dir.glob("sac_checkpoint_*_steps.zip")))
    if not zip_files:
        zip_files = sorted(list(checkpoint_dir.glob("*.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No checkpoint .zip files found in {checkpoint_dir}")

    val_suite = get_validation_suite()
    print(f"\n[SAC SELECTION] Evaluating {len(zip_files)} checkpoints in {checkpoint_dir.name}...")
    results = []

    for zf in zip_files:
        try:
            res = evaluate_checkpoint_on_validation_suite(zf, val_suite, algo="sac")
            results.append(res)
            print(f"  Checkpoint: {zf.name:<32} | Score: {res['composite_score']:+8.2f} | Tumbles: {res['tumble_count']} | RMS: {res['mean_rms_deg']:.3f}° | Final: {res['mean_final_err_deg']:.3f}°")
        except Exception as e:
            print(f"  [ERROR] Failed to evaluate {zf.name}: {e}")

    df = pd.DataFrame(results)
    df = df.sort_values(by="composite_score", ascending=False).reset_index(drop=True)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[SAC SELECTION] Ranking saved to: {output_csv}")

    best_record = df.iloc[0].to_dict()
    return best_record, df
