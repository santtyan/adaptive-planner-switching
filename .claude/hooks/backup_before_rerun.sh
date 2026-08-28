#!/bin/bash
# Faz backup de CSV/modelo antes de train_*.py ou rerun_*.py rodarem via Bash.
# Motivo: rerun_h1_mixed.py ja sobrescreveu um CSV canonico de 1.500 trials
# sem aviso em 31/07 (revertido so porque estava commitado). Ver DEVELOPMENT_LOG.md.
cmd=$(jq -r '.tool_input.command // empty')
if [[ -z "$cmd" ]]; then
  exit 0
fi
if echo "$cmd" | grep -qE '(train_[a-zA-Z0-9_]*|rerun_[a-zA-Z0-9_]*)\.py'; then
  root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
  ts=$(date +%Y%m%d-%H%M%S)
  found=0
  for f in $(echo "$cmd" | grep -oE '[A-Za-z0-9_./-]+\.(csv|zip|pkl)'); do
    target="$root/$f"
    [[ "$f" == /* ]] && target="$f"
    if [[ -f "$target" ]]; then
      cp -p "$target" "${target}.bak-${ts}"
      echo "backup: ${target}.bak-${ts}" >&2
      found=1
    fi
  done
  if [[ "$found" == "0" ]]; then
    known="$root/results_abstract/h1_real_2d_mixed_pool.csv $root/results_abstract/h1_real_2d_validation.csv $root/results_abstract/h1_hysteresis_2d.csv $root/results_abstract/h1_oneshot_perstep_switches.csv $root/results_abstract/urban_grid_results.csv $root/results_abstract/multiagent_astar_real_vs_sac.csv"
    for target in $known; do
      script=$(echo "$cmd" | grep -oE '(train_[a-zA-Z0-9_]*|rerun_[a-zA-Z0-9_]*)\.py' | head -1)
      case "$script" in
        rerun_h1_mixed.py) target="$root/results_abstract/h1_real_2d_mixed_pool.csv" ;;
        rerun_h1_real.py) target="$root/results_abstract/h1_real_2d_validation.csv" ;;
        rerun_h1_hysteresis.py) target="$root/results_abstract/h1_hysteresis_2d.csv" ;;
        rerun_h1_oneshot_perstep.py) target="$root/results_abstract/h1_oneshot_perstep_switches.csv" ;;
        rerun_urban.py) target="$root/results_abstract/urban_grid_results.csv" ;;
        rerun_multiagent_astar_real.py) target="$root/results_abstract/multiagent_astar_real_vs_sac.csv" ;;
        *) continue ;;
      esac
      if [[ -f "$target" ]]; then
        cp -p "$target" "${target}.bak-${ts}"
        echo "backup (alvo canonico do script): ${target}.bak-${ts}" >&2
        found=1
      fi
      break
    done
  fi
  if [[ "$found" == "0" ]]; then
    echo "aviso: comando roda train_*/rerun_* mas nenhum alvo de backup identificado (nem explicito no comando, nem canonico conhecido)" >&2
  fi
fi
exit 0
