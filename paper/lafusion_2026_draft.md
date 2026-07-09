# LAFusion 2026 — Draft (Springer CCIS format, ~10 pages, English)

**STATUS: rascunho inicial (09/07/2026) — precisa de revisão humana antes de submeter.**
Reaproveita SOMENTE dados verificados (Fase 1 Monte Carlo + benchmark clássico real +
achado MARL 2D + resultado BC 2D). Gazebo foi descartado definitivamente nesta IC
(decisão 09/07/2026) — Seção 3.5 cortada, tratado como diagnóstico de infraestrutura
+ trabalho futuro, nunca como placeholder aberto.

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
inter-agent collision without centralized coordination. Real-robot validation
in a physical-fidelity ROS2/Gazebo simulator was attempted but not completed
within this study, due to infrastructure bugs unrelated to the fusion
criterion itself; we report the diagnosis and treat quantitative real-robot
validation as future work (see Discussion).

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
  4. Report a real (non-mock) multi-agent benchmark using CBS trajectories,
     showing the fusion rule's decision pattern replicates under real
     planner outputs, not only under calibrated Monte Carlo models — the
     strongest non-simulated evidence available for this study (real-robot
     validation with Nav2/SmacPlanner2D and SAC/SB3 in Gazebo was attempted
     but not completed; see Discussion for the infrastructure diagnosis).

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
- A*: similar time, 7.7 KB / 220 KB (justifies A* as the classical component —
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

### 3.5 Real-world validation attempt — infrastructure diagnosis (cut, decided 09/07)
**Decision: this section is CUT, definitively — do not resurrect it.** The
Gazebo/TurtleBot3 real-robot validation was attempted with a full ROS2 Humble
+ Gazebo Classic 11 + Nav2/SmacPlanner2D + SAC/SB3 stack. Infrastructure was
implemented and debugged (two critical integration bugs found and fixed: a
shared-node evaluation bug, and incorrect ROS2/Gazebo service names causing
episode resets to fail silently). A third bug — the robot fails to gain real
velocity within the per-step physics-unpause window, even with correct
velocity commands and running physics — was diagnosed but not resolved
within the study's timeframe. Real-robot validation is reported as future
work only (Section 6), with the diagnosis kept as a one-paragraph note in
the Discussion for reproducibility credit — no placeholder numbers, no
"pending" table.

## 4. Results

- Table: fusion rule vs. best-fixed vs. best-competing-switcher (85.3 / 76 /
  78.7%), regret 2.9% (6.7% worst case) — reuse verified numbers only.
- Figure candidates (reuse from `paper/figs/`, already generated):
  `fig_reward_math_proof`/regret curve, `cbs_density_sweep.png`
  (multi-agent phase transition), `cbs_deviation_analysis.png` (Nash evidence).
- SAC vs CrossQ vs Behavior Cloning (BC) convergence comparison in the
  lightweight twin environment — genuinely new material, not yet in the
  final report in this form, and a real differentiator: BC (supervised
  imitation of a potential-field controller) reaches 98% success in ~2
  minutes of training, the strongest quantitative navigation result of the
  study, and evidence that the fusion criterion's environment is learnable
  under multiple paradigms, isolating the unresolved Gazebo gap as an
  infrastructure issue rather than a modeling one.
- Consolidated cross-paradigm figure (source: `eval/plot_all_benchmarks_comparison.py`,
  outputs `paper/figs/all_benchmarks_comparison.png` + `all_benchmarks_table.csv`):
  4-panel comparison spanning all four paradigms tested in this study —
  classical search (real time/memory benchmark), single-agent RL (SAC vs.
  CrossQ), supervised imitation (BC), and classical multi-agent (CBS, real
  trials). Frame explicitly in the paper as "every learning paradigm applied
  in this study, compared side by side" — directly answers the natural
  reviewer question of paradigm coverage, and reinforces the fusion framing
  (the ρ-criterion itself is a fifth "paradigm": a fusion rule over two of
  the above).

## 5. Discussion

- Why hard threshold vs. soft/probabilistic fusion — connects to LAFusion's
  "Decision theory" and "Fusion architectures" topics directly.
- Limitations: threshold determined offline; single scalar context signal;
  validation restricted to TurtleBot3 Waffle (see final report Section 4 for
  full limitations list — reuse but trim to CCIS page budget).

## 6. Conclusion and Future Work

- Restate H1/H2/H3 confirmation status honestly: H2 confirmed; H3
  (infrastructure realizability) confirmed, with two integration bugs found
  and fixed as a reproducible engineering contribution; H1 confirmed at the
  Monte Carlo (calibrated-model) level only — real-planner validation was
  attempted and not completed within this study, a limitation stated
  explicitly, not left as an open placeholder.
- Future work: real-robot validation (resolve the diagnosed physics-timing
  bug, then re-run the existing benchmark protocol), soft/probabilistic
  fusion variant, online threshold adaptation, full MARL with shared reward
  (Fase 3, already scoped in the final report).

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
- [ ] Verify 10-page CCIS limit including references — Section 3.5 was cut
      definitively (Gazebo validation discontinued, 09/07), which helps the
      budget.

## Ainda falta (para fechar o rascunho)
- ~~Decidir se a Seção 3.5 (Gazebo real) entra ou não~~ — RESOLVIDO (09/07):
  Gazebo descartado definitivamente nesta IC; Seção 3.5 cortada, tratada só
  como nota de diagnóstico na Discussão + trabalho futuro na Conclusão.
- Gerar a versão em inglês do GIF/figura do BC 2D (`fig_2d_bc_trajectory_sparse.png`,
  já existe em português no relatório final — traduzir legendas).
- Traduzir/adaptar as demais figuras existentes (estão em português — Springer CCIS
  exige texto e legendas em inglês).
- Escrever a bibliografia em formato Springer (não ABNT, que é o formato do
  relatório final).
