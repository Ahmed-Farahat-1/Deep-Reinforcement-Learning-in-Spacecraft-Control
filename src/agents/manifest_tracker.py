"""
Execution Manifest Tracker for Phase 4G Autonomous Training.
Tracks algorithm, seed, target timesteps, completed timesteps, status,
checkpoints, replay buffers, timestamps, and recovery history.
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "phase_4g_execution_manifest.csv"


def init_or_load_manifest() -> pd.DataFrame:
    """Load existing manifest or initialize with all 6 required runs."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        try:
            return pd.read_csv(MANIFEST_PATH)
        except Exception:
            pass

    records = [
        {"algorithm": "TD3", "seed": 42, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
        {"algorithm": "TD3", "seed": 123, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
        {"algorithm": "TD3", "seed": 2026, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
        {"algorithm": "SAC", "seed": 42, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
        {"algorithm": "SAC", "seed": 123, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
        {"algorithm": "SAC", "seed": 2026, "target_timesteps": 500000, "completed_timesteps": 0, "status": "PENDING", "latest_checkpoint": "None", "latest_replay_buffer": "None", "last_update": datetime.now().isoformat(), "recovery_history": "Initialized"},
    ]
    df = pd.DataFrame(records)
    df.to_csv(MANIFEST_PATH, index=False)
    return df


def update_manifest(
    algorithm: str,
    seed: int,
    completed_timesteps: int,
    status: str,
    latest_checkpoint: Optional[str] = None,
    latest_replay_buffer: Optional[str] = None,
    recovery_note: Optional[str] = None,
):
    """Atomically update a specific run's record in the manifest."""
    df = init_or_load_manifest()
    mask = (df["algorithm"] == algorithm) & (df["seed"] == seed)
    if not mask.any():
        df = pd.concat([df, pd.DataFrame([{
            "algorithm": algorithm,
            "seed": seed,
            "target_timesteps": 500000,
            "completed_timesteps": completed_timesteps,
            "status": status,
            "latest_checkpoint": str(latest_checkpoint or "None"),
            "latest_replay_buffer": str(latest_replay_buffer or "None"),
            "last_update": datetime.now().isoformat(),
            "recovery_history": str(recovery_note or "None"),
        }])], ignore_index=True)
    else:
        idx = df[mask].index[0]
        df.loc[idx, "completed_timesteps"] = completed_timesteps
        df.loc[idx, "status"] = status
        if latest_checkpoint:
            df.loc[idx, "latest_checkpoint"] = str(latest_checkpoint)
        if latest_replay_buffer:
            df.loc[idx, "latest_replay_buffer"] = str(latest_replay_buffer)
        df.loc[idx, "last_update"] = datetime.now().isoformat()
        if recovery_note:
            prev = str(df.loc[idx, "recovery_history"])
            if prev and prev != "nan" and prev != "None":
                df.loc[idx, "recovery_history"] = f"{prev} | {recovery_note}"
            else:
                df.loc[idx, "recovery_history"] = str(recovery_note)

    # Atomic write via temp file
    temp_path = MANIFEST_PATH.with_suffix(".tmp")
    df.to_csv(temp_path, index=False)
    temp_path.replace(MANIFEST_PATH)
    print(f"[MANIFEST] Updated {algorithm} Seed {seed}: status={status}, steps={completed_timesteps:,}")
