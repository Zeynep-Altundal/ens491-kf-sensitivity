"""Train and test the reference KalmanNet on one of our benchmark datasets.

What this script does
---------------------
KalmanNet itself is not reimplemented here: the script imports the authors' reference code
(``KalmanNet_TSP``, architecture 2, kept outside this repository; set ``KNET_REPO`` if it is not in
``~/ens491/KalmanNet_TSP``) and only connects it to our data:

1. Loads one ``.npz`` file written by ``linear.py``, ``sinusoidal.py``, ``lorenz.py`` or ``nclt.py``.
2. Builds the model that KalmanNet is *given* (f and h as torch functions). This is the model the
   classical filters get, so under mismatch KalmanNet and the KF/EKF start from the same wrong model:
     - linear, NCLT: ``F_model``, ``H_model`` from the file
     - sinusoidal:   ``--given partial`` (default, mismatch) or ``full``
     - Lorenz:       ``--given J5`` (default) or ``J2`` (process mismatch); always ``H_model`` = I,
                     so the ``rotH1`` files are an observation mismatch
3. Converts the arrays to the reference layout (N, n, T), trains with the reference pipeline
   (training / validation split, best checkpoint on validation) and tests on the test split.
4. Runs the KF/EKF with the same given model on the same test split, for comparison.
5. Saves the KalmanNet estimates, per-trajectory errors and training curves to
   ``~/data/knet_outputs/<run name>/results.npz`` (with ``best-model.pt``) and appends one summary
   row to ``results/kalmannet_results.csv``.

Training settings default to those of the reference scripts: linear 4000 steps, lr 1e-4; all other
benchmarks 2000 steps, lr 1e-3; batch 30 (at most the number of training trajectories), weight
decay 1e-3. The reference code is run on the CPU (it fails on the GPU).

Usage (from the repository root)::

    python -m experiments.knet_data.kalmannet_run --data linear_rotH10
    python -m experiments.knet_data.kalmannet_run --data "sinusoidal_inv_r2_+00.00dB" --given partial
    python -m experiments.knet_data.kalmannet_run --data "lorenz_rotH1_inv_r2_+10.00dB" --steps 50   # quick test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.filter_family import lorenz as zl
from experiments.linear_kf.kf import kalman_filter

from . import nclt, sinusoidal
from .common import DATA_DIR, batched_ekf, load_dataset, mse_db, per_trajectory_mse
from .eskf_equivalence import nclt_noise, sinusoidal_model
from .lorenz import jacobian as lorenz_jacobian

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results" / "kalmannet_results.csv"
KNET_REPO = Path(os.environ.get("KNET_REPO", Path.home() / "ens491" / "KalmanNet_TSP"))
OUT_DIR = Path(os.environ.get("KNET_OUT_DIR", DATA_DIR.parent / "knet_outputs"))

# (steps, learning rate) of the reference scripts; batch 30 and weight decay 1e-3 for all.
DEFAULTS = {"linear": (4000, 1e-4), "sinusoidal": (2000, 1e-3), "lorenz": (2000, 1e-3), "nclt": (2000, 1e-3)}


# --------------------------------------------------------------------------------------------
# Reference code
# --------------------------------------------------------------------------------------------

def import_reference():
    """Import KalmanNet, the training pipeline and the default arguments from the reference repo."""
    if not (KNET_REPO / "KNet" / "KalmanNet_nn.py").exists():
        sys.exit(f"reference KalmanNet code not found in {KNET_REPO} (set KNET_REPO)")
    sys.path.insert(0, str(KNET_REPO))
    from KNet.KalmanNet_nn import KalmanNetNN
    from Pipelines.Pipeline_EKF import Pipeline_EKF
    from Simulations.config import general_settings

    # The reference loads its own checkpoints with torch.load; since PyTorch 2.6 that needs
    # weights_only=False for a full model. The files are the ones written by this run.
    original_load = torch.load

    def load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_load(*a, **kw)

    if "weights_only" in original_load.__code__.co_varnames:
        torch.load = load

    argv, sys.argv = sys.argv, sys.argv[:1]   # general_settings() parses the command line
    try:
        ref_args = general_settings()
    finally:
        sys.argv = argv
    return KalmanNetNN, Pipeline_EKF, ref_args


# --------------------------------------------------------------------------------------------
# Model given to KalmanNet, as batched torch functions on (B, m, 1)
# The reference pipeline saves the whole network with torch.save, so f and h must be picklable:
# module-level functions bound with functools.partial (no lambdas).
# --------------------------------------------------------------------------------------------

def lin_map(M, x):
    """x -> M x for a batch x of shape (B, k, 1)."""
    return torch.matmul(M.to(x.device), x)


def sin_f(p, x):
    return p["alpha"] * torch.sin(p["beta"] * x + p["phi"]) + p["delta"]


def sin_h(p, x):
    return p["a"] * (p["b"] * x + p["c"]) ** 2


def lorenz_f(J, x):
    """x_{t+1} = F(x) x with F(x) = I + sum_{j=1..J} (A(x) dt)^j / j!  (same as filter_family/lorenz.py)."""
    B = x.shape[0]
    A = torch.zeros(B, 3, 3, dtype=x.dtype, device=x.device)
    A[:, 0, 0], A[:, 0, 1] = -zl.SIGMA, zl.SIGMA
    A[:, 1, 0], A[:, 1, 1] = zl.RHO - x[:, 2, 0], -1.0
    A[:, 2, 0], A[:, 2, 2] = x[:, 1, 0], -zl.BETA
    M = A * zl.DT
    term = torch.eye(3, dtype=x.dtype, device=x.device).expand(B, 3, 3)
    Fx = term.clone()
    for j in range(1, J + 1):
        term = torch.bmm(term, M) / j
        Fx = Fx + term
    return torch.bmm(Fx, x)


def torch_model(arrays: dict, meta: dict, given: str):
    """f and h (torch, batched) and a short description of the model given to KalmanNet."""
    bench = meta["benchmark"]
    if bench in ("linear", "nclt"):
        F = torch.tensor(arrays["F_model"], dtype=torch.float32)
        H = torch.tensor(arrays["H_model"], dtype=torch.float32)
        return partial(lin_map, F), partial(lin_map, H), "F_model, H_model"
    if bench == "sinusoidal":
        p = sinusoidal.PARAMS[given]
        return partial(sin_f, p), partial(sin_h, p), f"{given} parameters"
    if bench == "lorenz":
        J = {"J5": zl.J_DATA, "J2": zl.J_MISMATCH}[given]
        H = torch.tensor(arrays["H_model"], dtype=torch.float32)
        return partial(lorenz_f, J), partial(lin_map, H), f"Taylor order J={J}, H_model"
    raise ValueError(bench)


# --------------------------------------------------------------------------------------------
# Classical baseline with the same given model (numpy)
# --------------------------------------------------------------------------------------------

def classical_baseline(arrays: dict, meta: dict, given: str) -> tuple[str, np.ndarray]:
    bench = meta["benchmark"]
    y = arrays["y_test"]
    if bench == "linear":
        F, H = arrays["F_model"], arrays["H_model"]
        x_hat = np.stack([kalman_filter(yy, F, H, arrays["Q"], arrays["R"], arrays["x0"],
                                        np.zeros_like(arrays["Q"])).estimates for yy in y])
        return "KF (given model, true Q, R)", x_hat
    if bench == "nclt":
        q2, r2 = nclt_noise()
        x_hat = np.stack([kalman_filter(yy, nclt.F, nclt.H, q2 * nclt.Q_STRUCT, r2 * np.eye(2), x0,
                                        np.zeros((4, 4))).estimates
                          for yy, x0 in zip(y, arrays["x0_test"])])
        return f"KF (q2={q2:g}, r2={r2:g} from nclt_check)", x_hat
    if bench == "sinusoidal":
        funcs = sinusoidal_model(sinusoidal.PARAMS[given])
        x_hat, _ = batched_ekf(y, *funcs, arrays["Q"], arrays["R"], arrays["x0"], np.zeros((2, 2)))
        return f"EKF ({given} model, true Q, R)", x_hat
    J = {"J5": zl.J_DATA, "J2": zl.J_MISMATCH}[given]
    f = zl.make_f(J)
    H = arrays["H_model"]
    H_b = np.broadcast_to(H, (y.shape[0],) + H.shape)
    with np.errstate(all="ignore"):
        x_hat, _ = batched_ekf(y, f, lorenz_jacobian(f), lambda x: x @ H.T, lambda x: H_b,
                               arrays["Q"], arrays["R"], arrays["x0"], zl.P0_SCALE * np.eye(3))
    return f"EKF (J={J}, H_model, true Q, R)", x_hat


# --------------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------------

def to_ref(a: np.ndarray):
    """(N, T, k) numpy -> (N, k, T) float32 torch, the layout of the reference code."""
    return torch.tensor(np.ascontiguousarray(a.transpose(0, 2, 1)), dtype=torch.float32)


def run(data: str, given: str, steps: int | None, batch: int, lr: float | None, wd: float, seed: int) -> dict:
    KalmanNetNN, Pipeline_EKF, args = import_reference()
    arrays, meta = load_dataset(DATA_DIR / f"{data}.npz")
    bench = meta["benchmark"]
    if bench == "sinusoidal" and given not in ("full", "partial"):
        given = "partial"
    if bench == "lorenz" and given not in ("J5", "J2"):
        given = "J5"
    if bench in ("linear", "nclt"):
        given = "model"

    torch.manual_seed(seed)
    random.seed(seed)       # the pipeline draws training batches with random.sample
    np.random.seed(seed)

    m, n = arrays["x_train"].shape[2], arrays["y_train"].shape[2]
    f, h, model_desc = torch_model(arrays, meta, given)
    random_init = bench == "nclt"   # NCLT: every window starts from its own ground-truth state
    x0 = torch.tensor(arrays["x0"], dtype=torch.float32).reshape(m, 1)
    sysmodel = SimpleNamespace(
        f=f, h=h, m=m, n=n, T=arrays["x_train"].shape[1], T_test=arrays["x_test"].shape[1],
        m1x_0=x0, m2x_0=torch.zeros(m, m),
        prior_Q=torch.eye(m), prior_Sigma=torch.zeros(m, m), prior_S=torch.eye(n),   # reference defaults
    )

    d_steps, d_lr = DEFAULTS[bench]
    args.use_cuda = False
    args.n_steps = steps or d_steps
    args.n_batch = min(batch, arrays["x_train"].shape[0])
    args.lr = lr or d_lr
    args.wd = wd
    args.randomLength = False
    args.CompositionLoss = False

    run_name = f"kalmannet_{data}_{given}"
    out = OUT_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)

    knet = KalmanNetNN()
    knet.NNBuild(sysmodel, args)
    n_params = sum(p.numel() for p in knet.parameters() if p.requires_grad)
    pipe = Pipeline_EKF(time.strftime("%m.%d.%y_%H:%M:%S"), str(out) + "/", "KalmanNet")
    pipe.setssModel(sysmodel)
    pipe.setModel(knet)
    pipe.setTrainingParams(args)

    tr_in, tr_tg = to_ref(arrays["y_train"]), to_ref(arrays["x_train"])
    cv_in, cv_tg = to_ref(arrays["y_val"]), to_ref(arrays["x_val"])
    te_in, te_tg = to_ref(arrays["y_test"]), to_ref(arrays["x_test"])
    inits = {}
    if random_init:
        inits = {s: torch.tensor(arrays[f"x0_{s}"], dtype=torch.float32).unsqueeze(-1) for s in ("train", "val", "test")}

    print(f"\n=== {run_name}: {bench}, given model: {model_desc}, {n_params} parameters, "
          f"{args.n_steps} steps, batch {args.n_batch}, lr {args.lr:g}, wd {args.wd:g} ===\n")
    t0 = time.time()
    cv_lin, cv_db, tr_lin, tr_db = pipe.NNTrain(
        sysmodel, cv_in, cv_tg, tr_in, tr_tg, str(out) + "/",
        randomInit=random_init, cv_init=inits.get("val"), train_init=inits.get("train"))
    train_time = time.time() - t0
    _, _, test_db, x_out, infer_time = pipe.NNTest(
        sysmodel, te_in, te_tg, str(out) + "/", randomInit=random_init, test_init=inits.get("test"))

    x_hat = x_out.detach().cpu().numpy().transpose(0, 2, 1).astype(float)
    x_true = arrays["x_test"]
    per_traj_db = 10 * np.log10(per_trajectory_mse(x_hat, x_true))
    base_name, base_hat = classical_baseline(arrays, meta, given)

    row = {
        "run": run_name, "benchmark": bench, "data": data, "given_model": model_desc,
        "steps": args.n_steps, "batch": args.n_batch, "lr": args.lr, "wd": args.wd, "seed": seed,
        "parameters": n_params, "best_val_db": round(float(min(cv_db)), 3),
        "best_val_step": int(np.argmin(np.asarray(cv_db))),
        "knet_test_db": round(mse_db(x_hat, x_true), 3), "knet_test_std_db": round(float(np.std(per_traj_db)), 3),
        "baseline": base_name, "baseline_test_db": round(mse_db(base_hat, x_true), 3),
        "train_time_s": round(train_time, 1), "test_time_s": round(float(infer_time), 2),
    }
    np.savez_compressed(
        out / "results.npz", x_hat=x_hat, per_traj_mse=per_trajectory_mse(x_hat, x_true),
        baseline_x_hat=base_hat, train_db=np.asarray(tr_db), val_db=np.asarray(cv_db),
        meta=json.dumps({**row, "dataset_meta": meta}),
    )
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nKalmanNet test MSE: {row['knet_test_db']:.2f} dB (std {row['knet_test_std_db']:.2f})")
    print(f"{base_name}: {row['baseline_test_db']:.2f} dB")
    print(f"best validation {row['best_val_db']:.2f} dB at step {row['best_val_step']}; "
          f"training {train_time / 60:.1f} min")
    print(f"saved {out / 'results.npz'}; appended {RESULTS_CSV}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="dataset file name without .npz, e.g. linear_rotH10")
    ap.add_argument("--given", default="", help="sinusoidal: partial|full; lorenz: J5|J2")
    ap.add_argument("--steps", type=int, default=None, help="training steps (default: reference value)")
    ap.add_argument("--batch", type=int, default=30)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.data, a.given, a.steps, a.batch, a.lr, a.wd, a.seed)


if __name__ == "__main__":
    main()