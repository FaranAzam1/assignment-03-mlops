"""Deliberately break (or restore) the ml-inference Deployment, for the failure demonstrations.

    python scripts/inject_k8s_failure.py bad-image    # ImagePullBackOff: image tag does not exist
    python scripts/inject_k8s_failure.py bad-probe    # Pods Running but never Ready: wrong probe path
    python scripts/inject_k8s_failure.py restore      # re-apply the manifests in kubernetes/

Needs kubectl pointed at the demo cluster. Only touches the 'mlops' namespace.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

NS, DEP = "mlops", "ml-inference"
ROOT = Path(__file__).resolve().parent.parent


def kubectl(*args: str) -> None:
    subprocess.run(["kubectl", *args], check=True)


def patch(ops: list[dict]) -> None:
    kubectl("-n", NS, "patch", "deployment", DEP, "--type=json", "-p", json.dumps(ops))


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "bad-image":
        patch([{"op": "replace", "path": "/spec/template/spec/containers/0/image",
                "value": "ml-inference:9.9.9"}])
    elif mode == "bad-probe":
        patch([{"op": "replace", "path": "/spec/template/spec/containers/0/readinessProbe/httpGet/path",
                "value": "/does-not-exist"}])
    elif mode == "restore":
        kubectl("apply", "-f", str(ROOT / "kubernetes"))
        kubectl("-n", NS, "rollout", "status", f"deployment/{DEP}", "--timeout=120s")
    else:
        print(__doc__)
        return 2
    print(f"done: {mode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
