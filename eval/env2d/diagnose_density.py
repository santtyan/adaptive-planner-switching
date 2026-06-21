"""
diagnose_density.py — Degradação do SAC single-robot em densidade crescente.

O modelo sac_2d_best foi treinado SÓ no mundo 'sparse'. Este script mede quanto
ele generaliza para 'dense' e 'very_dense', rodando N episódios por densidade e
registrando taxa de goal / colisão / timeout.

Saída:
  paper/figs/2d/fig_2d_degradation_singlerobot.png/pdf
  results_abstract/singlerobot_density_results.csv

Uso:
  python3 -m eval.env2d.diagnose_density --episodes 100
"""

import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.env2d.env_2d import WORLDS
from eval.env2d.visualize_2d import _rollout, load_model, FIGS, ROOT

WORLDS_ORDER = ["sparse", "dense", "very_dense"]
RHO = {"sparse": 0.05, "dense": 0.30, "very_dense": 0.50}


def evaluate(model, world, episodes):
    counts = {"goal": 0, "collision": 0, "timeout": 0}
    for s in range(episodes):
        _, oc = _rollout(model, world, s)
        counts[oc] += 1
    n = float(episodes)
    return {k: v / n for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()

    model = load_model(os.path.join(ROOT, "models", "sac_2d_best"))
    if model is None:
        sys.exit("Modelo sac_2d_best não encontrado.")

    rows, rates = [], {}
    for w in WORLDS_ORDER:
        r = evaluate(model, w, args.episodes)
        rates[w] = r
        rows.append((w, RHO[w], r["goal"], r["collision"], r["timeout"]))
        print(f"  {w:11s} ρ={RHO[w]:.2f}  goal={r['goal']:.0%}  "
              f"coll={r['collision']:.0%}  timeout={r['timeout']:.0%}")

    # ── CSV ───────────────────────────────────────────────────
    csv_dir = os.path.join(ROOT, "results_abstract")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "singlerobot_density_results.csv")
    with open(csv_path, "w") as f:
        f.write("world,rho,goal_rate,collision_rate,timeout_rate\n")
        for w, rho, g, c, t in rows:
            f.write(f"{w},{rho},{g:.4f},{c:.4f},{t:.4f}\n")
    print(f"  → {csv_path}")

    # ── Figura ────────────────────────────────────────────────
    rhos   = [RHO[w] for w in WORLDS_ORDER]
    goal   = [rates[w]["goal"] * 100 for w in WORLDS_ORDER]
    coll   = [rates[w]["collision"] * 100 for w in WORLDS_ORDER]
    tmo    = [rates[w]["timeout"] * 100 for w in WORLDS_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rhos, goal, "o-", color="#2E7D32", lw=2.5, ms=9, label="Goal (sucesso)")
    ax.plot(rhos, coll, "s--", color="#C62828", lw=2,  ms=8, label="Colisão")
    ax.plot(rhos, tmo,  "^:",  color="#F9A825", lw=2,  ms=8, label="Timeout")
    for x, y in zip(rhos, goal):
        ax.annotate(f"{y:.0f}%", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, color="#2E7D32")
    ax.axvline(0.30, color="gray", ls="-.", alpha=0.6)
    ax.text(0.305, 5, r"$\rho^*=0{,}30$", color="gray", fontsize=9)
    ax.set_xlabel(r"Densidade de obstáculos  $\rho$")
    ax.set_ylabel("Taxa (%)")
    ax.set_title("Degradação do SAC single-robot vs densidade\n"
                 "(modelo treinado só no 'sparse' — não generaliza)")
    ax.set_ylim(-3, 105)
    ax.grid(alpha=0.3); ax.legend(loc="center right")
    for w, x in zip(WORLDS_ORDER, rhos):
        ax.annotate(w, (x, -1), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=9, color="dimgray")
    fig.tight_layout()

    out = os.path.join(FIGS, "2d", "fig_2d_degradation_singlerobot")
    fig.savefig(out + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}.png/pdf")


if __name__ == "__main__":
    main()
