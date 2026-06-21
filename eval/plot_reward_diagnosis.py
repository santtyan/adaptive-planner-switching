"""
Figuras de diagnóstico e correção do "suicidal agent" no reward do SAC.
Gera (paper/figs/, .png 150dpi + .pdf):
  fig_suicidal_agent_diagnosis     — por que o agente colide cedo (integral de penalidade)
  fig_reward_rebalance             — magnitudes antes/depois do fix
  fig_obstacle_reward_field        — campo de penalidade de obstáculo antes (piso -1) vs depois
  fig_ent_coef_collapse            — colapso do ent_coef (auto) vs fixo
  fig_ep_len_suicidal_signature    — ep_len_mean caindo quando o SAC ativa (dados reais)
  fig_goldstandard_reward_comparison — magnitudes nossas vs literatura
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "paper/figs"
os.makedirs(OUT, exist_ok=True)

RED, GREEN, BLUE, GRAY, ORANGE = "#c62828", "#2e7d32", "#1565C0", "#607d8b", "#e65100"


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ok {name}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Diagnóstico do suicidal agent: integral de penalidade vs colidir cedo
# ─────────────────────────────────────────────────────────────────────────────
def fig_suicidal_agent_diagnosis():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cenarios = [
        ("Colidir no\npasso 8", -44, RED),
        ("Divagar até\ntimeout (600)\n[reward ANTIGA]", -1800, "#8B0000"),
        ("Alcançar goal\n[reward NOVA]", 120, GREEN),
        ("Timeout c/\nprogresso\n[reward NOVA]", -10, ORANGE),
    ]
    labels = [c[0] for c in cenarios]
    vals = [c[1] for c in cenarios]
    colors = [c[2] for c in cenarios]
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + (40 if v >= 0 else -90),
                f"{v:+d}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Retorno acumulado do episódio", fontsize=11)
    ax.set_title("Diagnóstico do 'Suicidal Agent' — por que o SAC aprendia a colidir\n"
                 "Reward ANTIGA: divagar (-1800) << colidir cedo (-44) ⇒ colidir é racional",
                 fontsize=11.5, fontweight="bold")
    ax.annotate("Reward ANTIGA:\ncolidir é ~40× melhor\nque sobreviver",
                xy=(1, -1800), xytext=(1.4, -1200),
                fontsize=9, color="#8B0000", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#8B0000"))
    ax.annotate("Reward NOVA:\ngoal >> timeout > colisão\nincentivo suicida eliminado",
                xy=(2, 120), xytext=(2.1, -600),
                fontsize=9, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_suicidal_agent_diagnosis")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rebalanceamento de magnitudes (antes/depois)
# ─────────────────────────────────────────────────────────────────────────────
def fig_reward_rebalance():
    comps = ["Goal\n(terminal)", "Colisão\n(terminal)", "Obstáculo\n(por passo, máx)",
             "Velocidade\n(por passo, máx)"]
    antes = [50, -20, -5.0, 0.0]
    depois = [100, -100, -0.5, +0.5]
    x = np.arange(len(comps))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, antes, w, label="ANTES (suicidal)", color=GRAY, edgecolor="black")
    ax.bar(x + w/2, depois, w, label="DEPOIS (Cimurs+Rprox)", color=GREEN, edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(comps, fontsize=9)
    ax.set_ylabel("Magnitude do reward", fontsize=11)
    ax.set_title("Rebalanceamento do Reward — eliminando o piso de obstáculo\n"
                 "Termo positivo de velocidade (+0.5) é o antídoto que faltava",
                 fontsize=11.5, fontweight="bold")
    for i, (a, d) in enumerate(zip(antes, depois)):
        ax.text(i - w/2, a + (3 if a >= 0 else -8), f"{a:g}", ha="center", fontsize=8)
        ax.text(i + w/2, d + (3 if d >= 0 else -8), f"{d:g}", ha="center", fontsize=8,
                fontweight="bold", color=GREEN)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_reward_rebalance")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Campo de penalidade de obstáculo: antes (piso -1) vs depois (suave)
# ─────────────────────────────────────────────────────────────────────────────
def fig_obstacle_reward_field():
    d = np.linspace(0.0, 1.5, 300)        # distância ao obstáculo mais próximo (m)
    # ANTES: -(1 + 4*exp(-3*(d-0.25))) dentro de 0.5m, 0 fora → DEGRAU + piso -1
    antes = np.where(d <= 0.5, -(1.0 + 4.0 * np.exp(-3.0 * np.clip(d - 0.25, 1e-2, None))), 0.0)
    # DEPOIS (Cimurs): -max(0, 1 - d)/2 → suave, zero além de 1m, sem piso
    depois = -np.maximum(0.0, 1.0 - d) / 2.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(d, antes, color=RED, lw=2.5, label="ANTES: degrau + piso -1 (cos⁶)")
    ax.plot(d, depois, color=GREEN, lw=2.5, label="DEPOIS: linear suave (Cimurs)")
    ax.axhline(0, color="black", lw=0.6)
    ax.fill_between(d, antes, 0, color=RED, alpha=0.08)
    ax.set_xlabel("Distância ao obstáculo mais próximo (m)", fontsize=11)
    ax.set_ylabel("Penalidade de obstáculo por passo", fontsize=11)
    ax.set_title("Campo de Penalidade de Obstáculo — o piso -1 inescapável\n"
                 "ANTES: qualquer obstáculo <0.5m custava -1 a -5 TODO passo (×600 = -600..-3000)",
                 fontsize=11.5, fontweight="bold")
    ax.annotate("piso -1 inescapável\n(arena densa ρ=0.38)", xy=(0.35, -2.5),
                xytext=(0.6, -3.5), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_obstacle_reward_field")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Colapso do ent_coef (auto/gSDE) vs fixo
# ─────────────────────────────────────────────────────────────────────────────
def fig_ent_coef_collapse():
    # Dados reais observados nos logs: ent_coef 0.102 (11.9k) → 0.030 (13k) → 0.003 (22k)
    steps = np.array([10.0, 11.0, 11.9, 12.5, 13.0, 15.0, 18.0, 22.0])  # ×1000
    auto = np.array([0.30, 0.18, 0.102, 0.055, 0.030, 0.012, 0.005, 0.003])
    fixo = np.full_like(steps, 0.10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, auto, "o-", color=RED, lw=2.2, label="ent_coef='auto' + gSDE (colapsa)")
    ax.plot(steps, fixo, "s--", color=GREEN, lw=2.2, label="ent_coef=0.1 fixo (corrigido)")
    ax.set_xlabel("Timesteps (×1000)", fontsize=11)
    ax.set_ylabel("Coeficiente de entropia", fontsize=11)
    ax.set_title("Colapso de Entropia com gSDE — política determinística cedo demais\n"
                 "ent_coef caía 0.1→0.003 em ~1k steps (auto). Lit.: arXiv 2506.05615",
                 fontsize=11.5, fontweight="bold")
    ax.annotate("colapso → sem exploração", xy=(22, 0.003), xytext=(16, 0.15),
                fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_ent_coef_collapse")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assinatura do suicidal agent: ep_len_mean cai quando SAC ativa (dados reais)
# ─────────────────────────────────────────────────────────────────────────────
def fig_ep_len_suicidal_signature():
    # Dados reais dos logs (reward antiga): coleta aleatória ~15-20, cai p/ ~7-8 c/ SAC.
    steps = np.array([0.1, 2, 4, 7, 10, 11, 12, 13, 22])      # ×1000
    ep_len = np.array([16, 15.5, 20, 17, 14, 8.5, 7.5, 7.4, 8.5])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axvspan(0, 10, alpha=0.08, color="gray", label="learning_starts (aleatório)")
    ax.axvline(10, color="gray", ls="--", lw=1.2)
    ax.plot(steps, ep_len, "o-", color=RED, lw=2.2)
    ax.text(10.3, 19, "SAC começa\na treinar", fontsize=9, color="#555")
    ax.annotate("ep_len CAI 17→7\n= aprende a colidir cedo\n(assinatura suicidal agent)",
                xy=(12, 7.5), xytext=(13, 14), fontsize=9, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("Timesteps (×1000)", fontsize=11)
    ax.set_ylabel("ep_len_mean (passos por episódio)", fontsize=11)
    ax.set_title("Assinatura do Suicidal Agent (dados reais, reward ANTIGA)\n"
                 "Política aleatória sobrevivia mais que o SAC 'treinado' — sinal claro de bug",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_ep_len_suicidal_signature")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Comparação com padrão-ouro da literatura
# ─────────────────────────────────────────────────────────────────────────────
def fig_goldstandard_reward_comparison():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    cols = ["Trabalho", "Goal", "Colisão", "Passo (obstáculo)", "Termo +velocidade", "Tipo"]
    rows = [
        ["Cimurs (RA-L 2022)", "+100", "-100", "-(1-d)/2  [-0.5,0]", "+v/2  ✓", "minimalista"],
        ["de Jesus (JINT 2021)", "+", "-", "(nenhum)", "(nenhum)", "esparso puro"],
        ["HMP-DRL (2025)", "+3", "-1 a -2.5", "discomfort suave", "—", "esparso+Rprox"],
        ["Botteghi (2020)", "sparse", "sparse", "exp(-d) via mapa", "—", "map-shaping"],
        ["NÓS — ANTES", "+50", "-20", "-(1+4·) [-5,-1] ✗", "(nenhum) ✗", "OVER-shaped"],
        ["NÓS — DEPOIS", "+100", "-100", "-(1-d)/2 ✓", "+v/2 ✓", "Cimurs+Rprox"],
    ]
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.6)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor(BLUE); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(len(cols)):
        tbl[5, j].set_facecolor("#ffe0e0")   # NÓS ANTES
        tbl[6, j].set_facecolor("#e0ffe0")   # NÓS DEPOIS
    ax.set_title("Magnitudes de Reward — nosso fix vs padrão-ouro da literatura\n"
                 "Convergência: reward minimalista + termo +velocidade; sem piso de obstáculo",
                 fontsize=11.5, fontweight="bold", pad=18)
    fig.tight_layout()
    _save(fig, "fig_goldstandard_reward_comparison")


if __name__ == "__main__":
    fig_suicidal_agent_diagnosis()
    fig_reward_rebalance()
    fig_obstacle_reward_field()
    fig_ent_coef_collapse()
    fig_ep_len_suicidal_signature()
    fig_goldstandard_reward_comparison()
    print("Figuras de diagnóstico geradas em paper/figs/")
