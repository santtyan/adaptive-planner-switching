# Adaptive Context-Based Planner Switching

Dynamic switching between RRT* and PPO planners based on environmental context.

## Quick Start
```bash
pip install -r requirements.txt
python src/experiments/run_experiments.py --trials 10
```

## Results Preview
- 94.3% success rate (adaptive) vs ~80% (fixed planners)
- Switching based on obstacle density
- Real-time performance (<50ms P95)

## Project Structure
- src/: Core implementation
- experiments/: Training and evaluation scripts  
- results/: Experimental data and figures
- docs/: Paper drafts and documentation
