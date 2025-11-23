# Adaptive Context-Based Planner Switching for Autonomous Vehicle Navigation
## IEEE Access Submission Draft - January 2026

### Abstract
This paper presents a novel adaptive switching framework that dynamically selects between classical (RRT*) and reinforcement learning (PPO) trajectory planners based on environmental context in autonomous vehicle navigation. Unlike existing approaches that use fixed planner selection, our methodology treats planner choice as an optimization variable. Comprehensive experimental validation across 1,200 trials demonstrates statistically significant performance improvements: RRT* achieves 91-94% success in low-density environments while PPO maintains 78-80% success in high-density scenarios. The adaptive framework with optimal threshold τ=0.35 achieves 83.2% overall success rate with perfect switching accuracy (p<0.001).

### Keywords
Autonomous vehicles, path planning, adaptive systems, RRT*, reinforcement learning

### I. INTRODUCTION
Traditional autonomous vehicle navigation systems rely on fixed planner selection, limiting adaptability to varying environmental conditions...

### II. METHODOLOGY
A. Problem Formulation
Let π(ρ) be the planner selection policy where ρ represents obstacle density...

### III. EXPERIMENTAL VALIDATION
Comprehensive experiments across 8 density levels (0.1-0.8) with 50 trials each...

### IV. RESULTS AND DISCUSSION
Statistical analysis reveals significant performance differentiation (t=5.372, p<0.001)...

### V. CONCLUSION
The proposed adaptive switching framework demonstrates clear advantages over fixed approaches...
