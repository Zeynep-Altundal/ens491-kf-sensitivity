"""Exploratory: diagnosis of the non-finite training loss of the reference KalmanNet code.

The reference repository (no licence) is not copied: its modules are imported at run time from the
clone given on the command line. The default Lorenz configuration with full information is built
as in its ``main_lor_DT.py`` (switch 'full'), the training data are generated in memory (no file is
written and no pickle is loaded), and one training batch is passed through the untrained network:

1. forward pass with a non-finiteness check after every time step and on every sub-module output;
2. the same forward and backward pass under ``torch.autograd.detect_anomaly()``;
3. the reference state-transition function evaluated term by term at the last finite state.

Usage (from the repository root, with PyTorch installed)::

    python -m experiments.filter_family.lorenz_knet_diagnosis <path to the reference clone>
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    clone = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(clone))
    os.chdir(clone)
    import torch

    torch.manual_seed(0)
    random.seed(0)
    import Simulations.config as config
    from KNet.KalmanNet_nn import KalmanNetNN
    from Simulations.Extended_sysmdl import SystemModel
    from Simulations.Lorenz_Atractor import parameters as P

    sys.argv = [sys.argv[0]]
    args = config.general_settings()
    args.N_E, args.N_CV, args.N_T, args.T, args.T_test = 1000, 100, 200, 100, 100
    args.use_cuda = False
    args.n_batch = 30
    r2 = torch.tensor([0.1])
    q2 = 10 ** (-20 / 10) * r2
    sm = SystemModel(P.f, q2[0] * P.Q_structure, P.hRotate, r2[0] * P.R_structure, args.T, args.T_test,
                     P.m, P.n)
    sm.InitSequence(P.m1x_0, P.m2x_0)
    sm.GenerateBatch(args, args.N_E, args.T, randomInit=False)
    data_finite = bool(torch.isfinite(sm.Input).all() and torch.isfinite(sm.Target).all())
    idx = random.sample(range(args.N_E), k=args.n_batch)
    y = sm.Input[idx]
    x = sm.Target[idx]

    def fresh_net():
        net = KalmanNetNN()
        net.NNBuild(sm, args)
        net.train()
        net.batch_size = args.n_batch
        net.init_hidden_KNet()
        net.InitSequence(sm.m1x_0.reshape(1, P.m, 1).repeat(args.n_batch, 1, 1), args.T)
        return net

    # 1. forward pass with checks (the network is built right after the data, without re-seeding;
    # its initial weights are kept so that the anomaly run starts from the same weights)
    import copy

    net = fresh_net()
    init_state = copy.deepcopy(net.state_dict())
    step = [None]
    first_modules: list[str] = []

    def hook(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(o).all() and name not in first_modules and step[0] is not None:
                first_modules.append(name)
        return h

    for name, mod in net.named_children():
        mod.register_forward_hook(hook(name))
    rows = []
    first_step, first_quantities, bad_rows, last_finite = None, [], [], None
    with torch.no_grad():
        for t in range(args.T):
            step[0] = t
            xp = net(torch.unsqueeze(y[:, :, t], 2))
            checks = {"m1x_prior": net.m1x_prior, "m1y": net.m1y, "KGain": net.KGain, "m1x_posterior": xp}
            bad = [k for k, v in checks.items() if not torch.isfinite(v).all()]
            finite_x = xp[torch.isfinite(xp)]
            rows.append({"step": t, "max_abs_posterior_finite_part": float(finite_x.abs().max()) if finite_x.numel() else float("nan"),
                         "max_abs_gain": float(net.KGain[torch.isfinite(net.KGain)].abs().max()),
                         "non_finite_quantities": ";".join(bad)})
            if bad:
                first_step, first_quantities = t, bad
                bad_rows = torch.unique(torch.nonzero(~torch.isfinite(xp))[:, 0]).tolist()
                break
            last_finite = xp.clone()

    # 2. anomaly detection on a fresh network with the same initial weights
    net2 = fresh_net()
    net2.load_state_dict(init_state)
    net2.InitSequence(sm.m1x_0.reshape(1, P.m, 1).repeat(args.n_batch, 1, 1), args.T)
    anomaly = ""
    out = torch.zeros(args.n_batch, P.m, args.T)
    with torch.autograd.detect_anomaly():
        for t in range(args.T):
            out[:, :, t] = torch.squeeze(net2(torch.unsqueeze(y[:, :, t], 2)), 2)
        loss = torch.nn.MSELoss(reduction="mean")(out, x)
        loss_finite = bool(torch.isfinite(loss))
        try:
            loss.backward()
            anomaly = "none"
        except RuntimeError as e:
            anomaly = str(e).splitlines()[0]

    # 3. term-by-term evaluation of the reference f at the last finite state of the affected row
    terms = []
    if first_step is not None and last_finite is not None:
        xs = last_finite[bad_rows[0]:bad_rows[0] + 1]
        BX = torch.zeros([1, P.m, P.m])
        BX[:, 1, 0] = -xs[:, 2, 0]
        BX[:, 2, 0] = xs[:, 1, 0]
        A = BX + P.C
        for j in range(1, P.J + 1):
            Mj = torch.matrix_power(A * P.delta_t, j) / math.factorial(j)
            fin = Mj[torch.isfinite(Mj)]
            terms.append({"taylor_order_j": j, "term_finite": bool(torch.isfinite(Mj).all()),
                          "max_abs_finite_entries": float(fin.abs().max()) if fin.numel() else float("nan")})
        state_norm = float(xs.abs().max())
    else:
        state_norm = float("nan")

    summary = [
        {"quantity": "training data finite", "value": data_finite},
        {"quantity": "first step with a non-finite value (0-based)", "value": first_step},
        {"quantity": "non-finite quantities at that step", "value": ";".join(first_quantities)},
        {"quantity": "affected batch rows", "value": ";".join(map(str, bad_rows))},
        {"quantity": "first sub-modules with non-finite output", "value": ";".join(first_modules)},
        {"quantity": "max abs state of the affected row at the previous step", "value": state_norm},
        {"quantity": "float32 maximum", "value": float(torch.finfo(torch.float32).max)},
        {"quantity": "first non-finite Taylor term (order j)",
         "value": next((t["taylor_order_j"] for t in terms if not t["term_finite"]), "")},
        {"quantity": "operation", "value": "torch.matrix_power(A*delta_t, j) in f (Simulations/Lorenz_Atractor/parameters.py:90)"},
        {"quantity": "batch loss finite (anomaly run)", "value": loss_finite},
        {"quantity": "anomaly detection message", "value": anomaly},
        {"quantity": "PyTorch version", "value": torch.__version__},
    ]
    import csv

    RESULTS_DIR.mkdir(exist_ok=True)
    for name, table in (("lorenz_knet_nan_diagnosis", summary), ("lorenz_knet_nan_steps", rows),
                        ("lorenz_knet_nan_taylor_terms", terms)):
        with (RESULTS_DIR / f"{name}.csv").open("w", newline="") as fh:
            keys = list(table[0]) if table else ["empty"]
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in table:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in summary:
        print(f"{r['quantity']}: {r['value']}")
    for r in terms:
        print(r)


if __name__ == "__main__":
    main()
