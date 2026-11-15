"""Generate the synthetic classification dataset. v1 = 1000 rows, v2 = 2000 rows (more data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

VERSIONS = {"v1": dict(n_samples=1000, random_state=42), "v2": dict(n_samples=2000, random_state=7)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("version", choices=VERSIONS)
    p.add_argument("--out", default="data/dataset.csv")
    a = p.parse_args()
    X, y = make_classification(n_features=12, n_informative=6, n_redundant=2, class_sep=1.2,
                               flip_y=0.03, **VERSIONS[a.version])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]).round(5)
    df["label"] = y
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, version {a.version})")


if __name__ == "__main__":
    main()
