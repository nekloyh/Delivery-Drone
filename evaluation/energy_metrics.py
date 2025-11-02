"""Energy consumption analysis for RL episodes.

Computes statistics on energy usage, jerk (control smoothness), and
efficiency metrics from episode logs.
"""

from __future__ import annotations

import os
import logging
from typing import Dict

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _read_log(path: str) -> pd.DataFrame:
    """Read episode log from various formats (CSV, JSON, Parquet)."""
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.csv':
        return pd.read_csv(path)
    if ext in ('.json', '.jsonl'):
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    if ext in ('.parquet', '.pq'):
        return pd.read_parquet(path)
    
    raise ValueError(f'Unsupported log format: {ext}')


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for a 1D array."""
    x = np.asarray(x, dtype=np.float64)
    
    if x.size == 0:
        return {k: float('nan') for k in ['mean', 'std', 'median', 'p95', 'min', 'max']}
    
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'median': float(np.median(x)),
        'p95': float(np.percentile(x, 95)),
        'min': float(np.min(x)),
        'max': float(np.max(x)),
    }


def compute_energy_metrics(log_file: str) -> None:
    """Compute and print energy metrics from episode log.
    
    Args:
        log_file: Path to episode log file
        
    Raises:
        ValueError: If required columns are missing
    """
    df = _read_log(log_file)

    required_cols = {
        'energy_j': 'Total energy per episode [J]',
        'mean_jerk': 'Mean jerk magnitude over episode',
        'max_jerk': 'Max jerk magnitude over episode',
        'time_s': 'Episode duration [s]',
        'success': 'Episode success flag',
    }
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Log is missing required columns: {missing}')

    LOGGER.info('Episodes: %d', len(df))
    
    if 'success' in df:
        try:
            succ_rate = float(df['success'].astype(float).mean())
            LOGGER.info('Success rate: %.2f%%', succ_rate * 100)
        except Exception as exc:
            LOGGER.warning('Could not compute success rate: %s', exc)

    energy_stats = _summary_stats(df['energy_j'].to_numpy())
    mean_jerk_stats = _summary_stats(df['mean_jerk'].to_numpy())
    max_jerk_stats = _summary_stats(df['max_jerk'].to_numpy())

    def fmt(title: str, stats: Dict[str, float]) -> None:
        LOGGER.info(
            '%s: mean=%.3f, std=%.3f, median=%.3f, p95=%.3f, min=%.3f, max=%.3f',
            title, stats["mean"], stats["std"], stats["median"], 
            stats["p95"], stats["min"], stats["max"]
        )

    fmt('Energy [J]', energy_stats)
    fmt('Mean jerk', mean_jerk_stats)
    fmt('Max jerk', max_jerk_stats)

    if 'time_s' in df.columns:
        e_rate = (df['energy_j'] / df['time_s'].clip(lower=1e-6)).to_numpy()
        fmt('Energy rate [J/s]', _summary_stats(e_rate))
