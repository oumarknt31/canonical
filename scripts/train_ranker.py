"""Step 4: train + evaluate the analogy-quality classifier on your labels (grouped CV)."""
import json

import _path  # noqa: F401
from canonical.ranker import evaluate

r = evaluate()
if "error" in r:
    print(r["error"])
else:
    print(f"n={r['n']} good={r['n_good']} groups={r['n_groups']} folds={r['folds']}")
    for k, v in sorted(r["auc"].items(), key=lambda kv: -kv[1]):
        lo, hi = r["auc_ci95"][k]
        print(f"  ROC-AUC {k:22} {v:.3f}  (95% CI {lo:.2f}-{hi:.2f})")
    print("LR coefficients (standardised):", json.dumps({k: round(v, 2) for k, v in r["lr_coef"].items()}))
