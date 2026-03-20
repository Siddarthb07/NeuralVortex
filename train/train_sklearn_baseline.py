#!/usr/bin/env python3
"""
Phase-2 smoke baseline (CPU): regress pooled flow statistics from (rpm, blades, pitch, v_inflow).

Not an FNO — proves HDF5 → features → supervised loop.RMSE logged to stdout.
Requires: scikit-learn (pip install scikit-learn).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.hdf5_loader import stack_xy  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--h5", type=Path, default=Path("data/smoke.h5"))
    p.add_argument("--out", type=Path, default=Path("runs/sklearn_baseline_metrics.json"))
    args = p.parse_args()

    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return 1

    X, Y = stack_xy(args.h5)
    n = len(X)
    if n < 2:
        print("Need at least 2 samples in HDF5 for this baseline.")
        return 1
    if n <= 4:
        X_train = X_val = X
        y_train = y_val = Y
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, Y, test_size=0.25, random_state=42
        )

    metrics = {}
    preds_val = []
    for dim in range(Y.shape[1]):
        mdl = GradientBoostingRegressor(max_depth=3, n_estimators=80, random_state=42)
        mdl.fit(X_train, y_train[:, dim])
        pred = mdl.predict(X_val)
        preds_val.append(pred)
        rmse = float(np.sqrt(mean_squared_error(y_val[:, dim], pred)))
        metrics[f"target_{dim}_rmse"] = rmse

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
