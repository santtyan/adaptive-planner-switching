# LAFusion 2026 — Draft (Springer CCIS format, ~10 pages, English)

**STATUS: rascunho inicial (09/07/2026, madrugada) — precisa de revisão humana antes de submeter.**
Reaproveita SOMENTE dados verificados (Fase 1 Monte Carlo + benchmark clássico real +
achado MARL 2D). NÃO inclui números da Fase 2 Gazebo (ainda não convergida) —
tratada honestamente como trabalho em andamento/futuro, coerente com o padrão já
usado no relatório final.

---

## Title (candidate)

**A ρ-Criterion for Contextual Decision Fusion Between Classical and Learned
Trajectory Planners in Autonomous Navigation**

*(alternative, shorter: "Contextual Fusion of Classical and Learned Planners via
Local Obstacle Density")*

Framing note: the paper reframes "adaptive planner switching" explicitly as a
**decision-fusion problem** — selecting, per-context, which of two heterogeneous
estimators (a deterministic planner and a learned policy) to trust — to align
directly with LAFusion's scope (Algorithms > "Contextual adaptation", "Decision
theory"; Applications > "Robotics", "Autonomous systems").

## Authors

Yan Santos Leite¹, Aldo André Diaz Salazar²
¹School of Electrical, Mechanical and Computing Engineering, UFG
²Institute of Informatics, UFG

## Abstract (draft, ~200 words — CCIS style)

Autonomous robots operating in heterogeneous environments face a fundamental
trade-off: deterministic planners (e.g., A*) guarantee optimality in open space
but degrade under high obstacle density, while learned policies (e.g., Soft
Actor-Critic) generalize better to cluttered regions at unnecessary
computational cost in free space. Most prior work treats planner selection as a
fixed design decision. We frame this problem as one of **contextual decision
fusion**: given a real-time, scalar estimate of local obstacle density ρ, which
of two heterogeneous estimators — a classical planner or a learned policy —
should be trusted at each instant? We propose the ρ-criterion, a fusion rule
π(ρ) = {classical if ρ < ρ*; learned if ρ ≥ ρ*}, and determine the fusion
threshold ρ*=0.30 through 1,500 Monte Carlo trials with statistically
calibrated planner models. The fused policy achieves 85.3% success rate,
outperforming the best fixed planner (76%) and the best competing switching
method (78.7%), with 2.9% average regret (6.7% worst case) relative to an
oracle fusion rule. We further show the criterion extends naturally to
multi-robot settings, where independent per-robot fusion decisions reduce
inter-agent collision without centralized coordination. [Placeholder: one
sentence on real-robot/Gazebo validation status once available.]

**Keywords:** decision fusion, contextual adaptation, trajectory planning,
reinforcement learning, autonomous navigation, robotics

---

## 1. Introduction

- Motivating dilemma (A* vs. SAC), same framing as Section 1.1 of the final
  report — but from the FIRST sentence, use fusion vocabulary: "selecting which
  of two heterogeneous estimators to trust, conditioned on context" instead of
  "choosing a planner."
- Gap in literature: prior work optimizes a single planner's weights (He et
  al. 2025) or uses static geographic switching rules (Sensors 2025); none
  treats planner selection as a *formal contextual fusion problem* with regret
  guarantees relative to an oracle fusion rule.
- Contributions (bullet list, CCIS style):
  1. Formalize adaptive planner selection as a contextual decision-fusion
     problem with an oracle-relative regret bound.
  2. Empirically determine the fusion threshold ρ* via 1,500 calibrated Monte
     Carlo trials, with regret ≤5% (H2, confirmed 2.9% avg / 6.7% worst case).
  3. Extend the fusion rule to a decentralized multi-robot setting (Dec-POMDP
     formulation) and show empirical evidence of a Nash-candidate equilibrium.
  4. [If Fase 2 lands in time] Validate the fusion rule with real classical
     and learned planners (Nav2/SmacPlanner2D, SAC/SB3) on a physical-fidelity
     ROS2/Gazebo simulation.

## 2. Related Work

- Classical planning: Dijkstra, A*, Hart et al. 1968.
- Learned navigation policies: SAC (Haarnoja 2018), Cimurs (RA-L 2022),
  de Jesus (2021) — reward shaping for sparse-reward navigation.
- Hybrid/switching approaches: He et al. 2025 (single-planner weight
  optimization — NOT a fusion rule), Sensors 2025 (static geographic rule).
- Position this work relative to classical **decision fusion** literature
  (Dempster-Shafer, Bayesian fusion, ensemble methods) — even though the
  ρ-criterion is a hard/deterministic threshold rule rather than a
  probabilistic fusion operator, framing it against that literature strengthens
  the LAFusion fit. Possible extension to discuss as future work: soft fusion
  (weighted blend near ρ*) vs. the current hard switch.

## 3. Methodology

### 3.1 Problem formulation
- State: local obstacle density ρ ∈ [0,1] (context signal for fusion).
- Two "estimators": classical planner π_A* (optimal, high cost under density)
  and learned policy π_SAC (robust, unnecessary cost in open space).
- Fusion rule: π(ρ) = π_A* if ρ<ρ*, else π_SAC. Deterministic hard fusion by design
  (rationale: real-time constraint — soft/probabilistic blending would require
  per-step arbitration cost incompatible with the control loop rate).

### 3.2 Classical planner benchmark (real, not simulated)
Reuse the classical algorithm benchmark from the final report — REAL,
optimized implementations (heapq for Dijkstra/A*, dense matrix for
Floyd-Warshall, Bellman-Ford re-weighting for Johnson):
- Dijkstra: 0.07 ms / 3.7 KB (100 nodes) → 2.46 ms / 85 KB (2,500 nodes).
- A*: similar time, 6.6 KB / 220 KB (justifies A* as the classical component —
  best time/memory scaling among all four).
- Floyd-Warshall: 39 s / 22 MB on a 30×30 grid — infeasible for real-time use.
- Table + O(V log V) vs O(V³) discussion — reuse Section 2.1 of the report.

### 3.3 Fusion threshold determination (Monte Carlo)
- 1,500 trials, calibrated planner models (statistically reproduce published
  success rates from He et al. 2025 / Sensors 2025 under matched density
  conditions — declare explicitly as a modeling choice, not a limitation
  hidden from reviewers).
- Regret(π) = E[R_oracle] − E[R_π]; oracle = best planner per trial.
- Result: ρ*=0.30 minimizes regret; 2.9% average, 6.7% worst case.

### 3.4 Multi-robot extension (decentralized fusion)
- Dec-POMDP formulation: each robot i observes local ρ_i, applies π(ρ_i)
  independently — O(1) decision per agent, no centralized coordination needed.
- Deviation experiment: empirical evidence toward a Nash-candidate equilibrium
  (deviating from ρ-criterion costs +2.1% if always-A*, +14.1% if always-SAC).
- CBS-based validation (2/3/5 agents), density sweep confirms ρ*≈0.28-0.32
  transition holds at the multi-agent level too.

### 3.5 [Conditional] Real-world validation
- ROS2 Humble + Gazebo Classic + TurtleBot3 Waffle, Nav2/SmacPlanner2D (C++)
  vs. SAC/Stable-Baselines3. **Only include this section with real numbers.**
  If not ready by submission, state as future work in Section 6, do NOT use
  placeholders in a submitted paper.

## 4. Results

- Table: fusion rule vs. best-fixed vs. best-competing-switcher (85.3 / 76 /
  78.7%), regret 2.9% (6.7% worst case) — reuse verified numbers only.
- Figure candidates (reuse from `paper/figs/`, already generated):
  `fig_reward_math_proof`/regret curve, `cbs_density_sweep.png`
  (multi-agent phase transition), `cbs_deviation_analysis.png` (Nash evidence).
- [NEW, if 2D comparison matrix finishes tonight] SAC vs CrossQ vs BC
  convergence comparison in the lightweight twin environment — genuinely new
  material not yet in the final report, could differentiate this paper.

## 5. Discussion

- Why hard threshold vs. soft/probabilistic fusion — connects to LAFusion's
  "Decision theory" and "Fusion architectures" topics directly.
- Limitations: threshold determined offline; single scalar context signal;
  validation restricted to TurtleBot3 Waffle (see final report Section 4 for
  full limitations list — reuse but trim to CCIS page budget).

## 6. Conclusion and Future Work

- Restate H1/H2/H3 confirmation status honestly (H1/H2 confirmed at Fase 1
  level; H3 infrastructure confirmed; Fase 2 quantitative validation ongoing).
- Future work: soft/probabilistic fusion variant, online threshold adaptation,
  full MARL with shared reward (Fase 3, already scoped in the final report).

---

## Award-quality checklist (Yan's ask: "quero ganhar um prêmio")
- [ ] Abstract must be self-contained and precise — no vague claims, every
      number traceable to a real experiment (done in draft above — verify
      once more before submission).
- [ ] Explicitly cite the fusion-theory framing in intro AND conclusion (novelty
      angle for a fusion-themed venue — this is the differentiator vs. a
      generic robotics paper).
- [ ] Add a comparison table against He et al. 2025 / Sensors 2025 method
      characteristics (not just numbers) — reviewers reward explicit
      positioning against related work.
- [ ] All figures must be regenerated at print resolution (300dpi) for the
      Springer template — check `paper/figs/CATALOG.md` for existing assets.
- [ ] Have Prof. Aldo review before submission (co-author, INF/UFG).
- [ ] Grammar/English pass — this draft is written directly in English but
      needs a native-level polish pass before submission.
- [ ] Verify 10-page CCIS limit including references — current section list is
      ambitious, likely needs trimming (Section 3.5 is the first candidate to
      cut/shrink if real Gazebo data isn't ready).

## Ainda falta (para fechar o rascunho)
- Decidir se a Seção 3.5 (Gazebo real) entra ou não, dependendo do platô ser
  resolvido a tempo (ver [[project-treino-sparse-08jul]]).
- Traduzir/adaptar as figuras existentes (estão em português — Springer CCIS
  exige texto e legendas em inglês).
- Escrever a bibliografia em formato Springer (não ABNT, que é o formato do
  relatório final).
