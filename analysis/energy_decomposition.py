from __future__ import annotations

import csv
import os
import numpy as np
import pyarrow.parquet as pq

from _common import ROOT

TURNS_CSV    = os.path.join(ROOT, "data", "turns.csv")
PARQUET_A100 = os.path.join(
    ROOT, "data", "energy_decomposition", "parquet",
    "energy_decomp_a100_a3b__turns.parquet",
)
OUT_MAIN  = os.path.join(ROOT, "tables", "data", "energy_decomposition.csv")
OUT_DIAG  = os.path.join(ROOT, "tables", "data", "energy_decomposition_diagnostics.csv")
BOOT_N    = 500

KEEP_AGENTS = {"VanillaRunner", "BOADMASRunner", "HyperAgentMASRunner"}

SETUPS = [
    ("A3B / A100", "parquet", PARQUET_A100, None),
    ("A3B / B200", "csv",     "Qwen/Qwen3.6-35B-A3B-FP8", "cache_h_delta"),
    ("27B / B200", "csv",     "Qwen/Qwen3.6-27B-FP8",     "cached_tokens"),
]


def _design_csv(model_filter: str, cached_col: str):
    Xu_l, Xc_l, Y_l, t_l, E_l, ep_l = [], [], [], [], [], []
    with open(TURNS_CSV) as f:
        for r in csv.DictReader(f):
            if r["model"] != model_filter or r["agent_class"] not in KEEP_AGENTS:
                continue
            try:
                xc  = float(r[cached_col])
                xin = float(r["input_tokens"])
                xu  = xin - xc
                y   = float(r["output_tokens"])
                e   = float(r["joules"])
                t   = float(r["duration_seconds"])
            except (ValueError, KeyError):
                continue
            Xu_l.append(xu); Xc_l.append(xc); Y_l.append(y)
            t_l.append(t);   E_l.append(e)
            ep_l.append(f"{r['task_id']}__{r['agent_class']}__seed{r['seed']}")
    Xu = np.array(Xu_l); Xc = np.array(Xc_l); Y = np.array(Y_l)
    tt = np.array(t_l);  E  = np.array(E_l)
    eps = np.array(ep_l)
    keep = ~(np.isnan(E) | np.isnan(tt))
    Xu, Xc, Y, tt, E, eps = (a[keep] for a in (Xu, Xc, Y, tt, E, eps))
    y_mJ = E * 1000.0
    X = np.column_stack([np.ones_like(Xu), Xu, Xc, Y])
    _, ep_idx = np.unique(eps, return_inverse=True)
    return X, y_mJ, ep_idx


def _design_parquet(path: str):
    t = pq.read_table(path)
    Xu  = np.asarray(t["Xu"],     dtype=float)
    Xc  = np.asarray(t["Xc"],     dtype=float)
    Y   = np.asarray(t["Y"],      dtype=float)
    tt  = np.asarray(t["t"],      dtype=float)
    E   = np.asarray(t["E"],      dtype=float)
    iw  = np.asarray(t["idle_W"], dtype=float)
    eps = np.asarray(t["episode_id"])
    keep = ~(np.isnan(iw) | np.isnan(E) | np.isnan(tt))
    Xu, Xc, Y, tt, E, iw, eps = (a[keep] for a in (Xu, Xc, Y, tt, E, iw, eps))
    y_mJ = (E - iw * tt) * 1000.0
    X = np.column_stack([np.ones_like(Xu), Xu, Xc, Y])
    _, ep_idx = np.unique(eps, return_inverse=True)
    return X, y_mJ, ep_idx


def _fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def _r2(y, yhat):
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss_tot if ss_tot else float("nan")


def _partial_r2(X, y, j):
    other = np.delete(X, j, axis=1)
    b_y,  *_ = np.linalg.lstsq(other, y,       rcond=None)
    b_xj, *_ = np.linalg.lstsq(other, X[:, j], rcond=None)
    e_y  = y       - other @ b_y
    e_xj = X[:, j] - other @ b_xj
    if (e_xj ** 2).sum() == 0:
        return 0.0
    beta_j = float(e_y @ e_xj / (e_xj @ e_xj))
    return _r2(e_y, beta_j * e_xj)


def _vif(X, j):
    other = np.delete(X, j, axis=1)
    b, *_ = np.linalg.lstsq(other, X[:, j], rcond=None)
    yhat = other @ b
    r2 = _r2(X[:, j], yhat)
    if r2 != r2:
        return float("nan")
    return float(1.0 / (1.0 - r2)) if r2 < 1.0 - 1e-12 else float("inf")


def _cluster_bootstrap(X, y, ep_idx, B, rng):
    n_ep = int(ep_idx.max()) + 1
    p = X.shape[1]
    XtX_ep = np.zeros((n_ep, p, p))
    Xty_ep = np.zeros((n_ep, p))
    for e in range(n_ep):
        mask = ep_idx == e
        Xe, ye = X[mask], y[mask]
        XtX_ep[e] = Xe.T @ Xe
        Xty_ep[e] = Xe.T @ ye
    samples = np.empty((B, p))
    for b in range(B):
        sel = rng.integers(0, n_ep, size=n_ep)
        samples[b] = np.linalg.solve(
            XtX_ep[sel].sum(axis=0),
            Xty_ep[sel].sum(axis=0),
        )
    return samples


def _split_half_cluster(X, y, ep_idx, rng, n_repeats=10):
    n_ep = int(ep_idx.max()) + 1
    deltas = []
    for _ in range(n_repeats):
        perm = rng.permutation(n_ep)
        h = n_ep // 2
        m1 = np.isin(ep_idx, perm[:h])
        m2 = np.isin(ep_idx, perm[h:])
        b1 = _fit(X[m1], y[m1])
        b2 = _fit(X[m2], y[m2])
        deltas.append(np.abs(b1 - b2) / (0.5 * np.abs(b1 + b2) + 1e-9))
    return np.mean(deltas, axis=0)


def _compute(label: str, X, y, ep_idx, rng):
    n, _ = X.shape
    n_ep = int(ep_idx.max()) + 1
    beta = _fit(X, y)
    boot = _cluster_bootstrap(X, y, ep_idx, BOOT_N, rng)
    se   = boot.std(axis=0, ddof=1)
    ci   = np.percentile(boot, [2.5, 97.5], axis=0)
    partial = np.array([_partial_r2(X, y, j) for j in range(X.shape[1])])
    vif     = np.array([_vif(X, j)         for j in range(X.shape[1])])
    split   = _split_half_cluster(X, y, ep_idx, rng)
    r2      = _r2(y, X @ beta)
    corr_e  = np.array([
        np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else float("nan")
        for j in range(X.shape[1])
    ])
    return {
        "label":   label,
        "n_turn":  n,
        "n_ep":    n_ep,
        "beta":    beta,
        "se":      se,
        "ci":      ci,
        "partial": partial,
        "vif":     vif,
        "split":   split,
        "r2":      r2,
        "corr_e":  corr_e,
    }


def _write_main(results, path: str) -> None:
    fields = ["model", "gpu", "bu", "su", "bc", "sc", "bo", "so", "r"]
    out_rows = []
    for r in results:
        _, bu, bc, bo = r["beta"]
        _, su, sc, so = r["se"]
        model_part, gpu_part = [s.strip() for s in r["label"].split("/", 1)]
        bo_s = f"{bo:,.0f}".replace(",", "{,}")
        so_int = f"{so:.0f}"
        so_cell = ("\\phantom{0}" if len(so_int) < 2 else "") + f"\\mathbf{{{so_int}}}"
        out_rows.append([
            model_part,
            gpu_part,
            f"${bu:.2f}$",
            f"${su:.2f}$",
            f"${bc:.2f}$",
            f"${sc:.2f}$",
            f"{{$\\mathbf{{{bo_s}}}$}}",
            f"${so_cell}$",
            f"${r['r2']:.3f}$",
        ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for row in out_rows:
            f.write(",".join(row) + "\n")
    print(f"wrote {path}")


def _write_diag(results, path: str) -> None:
    coef_names_tex = [r"\beta_u", r"\beta_c", r"\beta_o"]
    coef_idx = [1, 2, 3]
    fields = ["block", "lab", "coef", "r", "vif", "pr"]
    out_rows = []
    for r in results:
        for k, j in enumerate(coef_idx):
            label_cell = (
                f"\\multirow{{3}}{{*}}{{{r['label']} (R$^2$={r['r2']:.3f})}}"
                if k == 0 else ""
            )
            out_rows.append([
                r["label"],
                label_cell,
                f"${coef_names_tex[k]}$",
                f"${r['corr_e'][j]:.3f}$",
                f"${r['vif'][j]:.2f}$",
                f"${r['partial'][j]:.4f}$",
            ])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(fields) + "\n")
        for row in out_rows:
            f.write(",".join(row) + "\n")
    print(f"wrote {path}")


def main() -> None:
    rng = np.random.default_rng(0)
    results = []
    for entry in SETUPS:
        label, kind = entry[0], entry[1]
        if kind == "csv":
            _, _, model_filter, cached_col = entry
            X, y, ep_idx = _design_csv(model_filter, cached_col)
        else:
            _, _, path, _ = entry
            X, y, ep_idx = _design_parquet(path)
        results.append(_compute(label, X, y, ep_idx, rng))

    print(f"\n{'Setup':14s} {'n_turn':>7s} {'n_ep':>5s} "
          f"{'beta_u':>16s} {'beta_c':>14s} {'beta_o':>20s} "
          f"{'b_o/b_u':>9s} {'R^2':>7s}")
    for r in results:
        _, bu, bc, bo = r["beta"]
        _, su, sc, so = r["se"]
        print(
            f"{r['label']:14s} {r['n_turn']:7d} {r['n_ep']:5d} "
            f"{bu:7.2f} ± {su:5.2f}  "
            f"{bc:6.2f} ± {sc:4.2f}  "
            f"{bo:8.1f} ± {so:6.1f}  "
            f"{bo/bu:7.0f}x {r['r2']:7.4f}"
        )

    _write_main(results, OUT_MAIN)
    _write_diag(results, OUT_DIAG)


if __name__ == "__main__":
    main()
