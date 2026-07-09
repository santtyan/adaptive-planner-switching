"""
Train SAC on TurtleBot3GazeboEnv.

SAC with automatic entropy tuning (ent_coef='auto') is preferred over DDPG
for continuous control — more sample-efficient and robust (Haarnoja 2018).

Usage (headless, recommended):
    TURTLEBOT3_MODEL=waffle gzserver worlds/dense_custom.world &
    ros2 launch turtlebot3_bringup robot.launch.py use_sim_time:=True &
    python3 train_sac.py --steps 500000 --seed 42

Parallel training (2 instances, separate ROS_DOMAIN_ID):
    ROS_DOMAIN_ID=0 python3 train_sac.py --seed 42 &
    ROS_DOMAIN_ID=1 python3 train_sac.py --seed 43 &

Plan B: if ep_rew_mean does not cross 50 by step 300k, abort and use PPO only.

Outputs:
    models/sac_<seed>_<steps>.zip        best model
    models/sac_<seed>_<steps>_final.zip  final model
    logs/sac_<seed>/                     TensorBoard logs
"""

import argparse
import glob
import os
import re

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from turtlebot3_gym_env.gazebo_gym_env import TurtleBot3GazeboEnv, _GazeboEnvNode


def find_latest_checkpoint(models_dir: str, seed: int):
    """Localiza o checkpoint mais recente (por DATA DE MODIFICAÇÃO) salvo por
    CheckpointCallback.

    Retorna (model_path, replay_buffer_path_ou_None, vecnormalize_path_ou_None)
    ou None se nenhum checkpoint existir. Usado para auto-resume: se o container
    morrer (crash, reboot, docker stop) e for reiniciado, o treino continua do
    último checkpoint em vez de do zero — ver [[project-treino-sparse-08jul]].

    IMPORTANTE: seleciona por mtime (data de modificação), NÃO pelo maior número
    de step no nome do arquivo. Runs antigos abandonados (ex.: 21/06, antes do
    fix "atrator ocioso") podem ter checkpoints com step NUMERICAMENTE MAIOR que
    o run atual mas muito mais antigos — escolher por step teria retomado do
    checkpoint errado (achado nesta sessão, 08/07, antes de aplicar em produção).
    """
    pattern = os.path.join(models_dir, f"sac_{seed}_ckpt_*_steps.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    best = max(candidates, key=os.path.getmtime)

    def step_of(path: str) -> int:
        m = re.search(r"_(\d+)_steps\.zip$", path)
        return int(m.group(1)) if m else -1

    step = step_of(best)
    replay_path = os.path.join(
        models_dir, f"sac_{seed}_ckpt_replay_buffer_{step}_steps.pkl"
    )
    vecnorm_path = os.path.join(
        models_dir, f"sac_{seed}_ckpt_vecnormalize_{step}_steps.pkl"
    )
    return (
        best,
        replay_path if os.path.isfile(replay_path) else None,
        vecnorm_path if os.path.isfile(vecnorm_path) else None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--fresh", action="store_true",
                   help="Ignora checkpoints existentes e treina do zero "
                        "(padrão é AUTO-RESUME do checkpoint mais recente)")
    # Exploração (08/07/2026): ent_coef=0.1 FIXO trava std no valor de
    # inicialização (~exp(-3)=0.0498) porque remove o mecanismo de correção do
    # SAC contra colapso de entropia (que só existe com "auto" + target_entropy).
    # Ver [[project-treino-sparse-08jul]] para o diagnóstico completo.
    p.add_argument("--ent-coef", default="0.1",
                   help="'auto' restaura a correção automática de entropia; "
                        "valor fixo (ex. '0.1') não reage à entropia atual")
    p.add_argument("--target-entropy", type=float, default=None,
                   help="Só usado com --ent-coef auto. Default SB3 é -dim(ação) "
                        "= -2 (pode ser agressivo demais); -1.0 é mais permissivo")
    p.add_argument("--reset-log-std-on-resume", action="store_true",
                   help="Ao retomar de checkpoint, reinicializa só a camada "
                        "log_std do ator (mantém crítico + resto do ator) — "
                        "destrava exploração congelada sem perder o treino")
    # Early abort (Plan B) — DESLIGADO por padrão (20/06/2026).
    # O threshold antigo (ep_rew_mean<50 @ 300k) era inatingível em mundo denso
    # (R_GOAL=50 exigiria quase todo episódio fechando goal) → matava o run.
    # Ativar explicitamente com --planb-enable se desejar o early-abort.
    p.add_argument("--planb-enable", action="store_true",
                   help="Habilita o early-abort Plan B (desligado por padrão)")
    p.add_argument("--planb-step", type=int, default=300_000,
                   help="Check ep_rew_mean at this step; abort if < planb-threshold")
    p.add_argument("--planb-threshold", type=float, default=50.0)
    # Early-stop por sucesso (corta horas pós-convergência; 500k é teto, não alvo).
    p.add_argument("--stop-success-rate", type=float, default=0.85,
                   help="Encerra se success_rate ≥ isto no goal mais distante")
    p.add_argument("--stop-patience", type=int, default=3,
                   help="Checagens consecutivas acima do limiar p/ encerrar")
    return p.parse_args()


class PlanBCallback(BaseCallback):
    """Stop training early if SAC has not converged by planb_step.

    BaseCallback subclass (não um callable solto) para ser compatível com
    CallbackList quando passado em lista junto a EvalCallback/CheckpointCallback.
    """

    def __init__(self, check_step: int, threshold: float, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._check_step = check_step
        self._threshold = threshold
        self._triggered = False

    def _on_step(self) -> bool:
        if self._triggered:
            return True
        if self.num_timesteps >= self._check_step:
            mean_reward = self.logger.name_to_value.get(
                "rollout/ep_rew_mean", float("inf")
            )
            if mean_reward < self._threshold:
                print(
                    f"\n[PlanB] ep_rew_mean={mean_reward:.1f} < {self._threshold} "
                    f"at step {self.num_timesteps}. Aborting SAC — use PPO only.\n"
                )
                self._triggered = True
                return False  # stops training
        return True


class StopOnSuccessCallback(BaseCallback):
    """Encerra o treino quando o agente domina o goal mais distante do curriculum.

    Motivação (20/06/2026): 500k steps é um TETO de segurança, não um alvo. Com a
    arena 4×4m + curriculum até 3.0m + gradient_steps=4, a convergência esperada é
    ~150-250k. Rodar 500k cegamente desperdiça horas após o platô de sucesso.

    Critério: success_rate ≥ `threshold` na janela do curriculum, COM
    `at_max_curriculum=True` (curr_max_dist já no máximo), sustentado por
    `patience` checagens consecutivas. Lê de info[] exposto pelo env.
    """

    def __init__(self, threshold: float = 0.85, patience: int = 3,
                 check_freq: int = 2_000, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._threshold = threshold
        self._patience = patience
        self._check_freq = check_freq
        self._hits = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self._check_freq != 0:
            return True  # gate rápido — só lê infos em check_freq
        infos = self.locals.get("infos")
        if not infos:
            return True
        info = infos[0]
        at_max = info.get("at_max_curriculum", False)
        sr = info.get("success_rate", 0.0)
        if at_max and sr >= self._threshold:
            self._hits += 1
            if self.verbose:
                print(f"[StopOnSuccess] sr={sr:.0%} @ max-curriculum "
                      f"({self._hits}/{self._patience}) step {self.num_timesteps}")
            if self._hits >= self._patience:
                print(f"\n[StopOnSuccess] CONVERGIU — sr={sr:.0%} sustentado. "
                      f"Encerrando em {self.num_timesteps} steps (teto era {self.locals.get('total_timesteps','?')}).\n")
                return False
        else:
            self._hits = 0
        return True


class BestRolloutModelCallback(BaseCallback):
    """Salva o melhor modelo pela média de recompensa dos episódios recentes.

    Substitui o EvalCallback: o env de avaliação compartilhava o mesmo
    _GazeboEnvNode do treino (um único publisher /cmd_vel e subscriber /scan),
    então a eval corrompia o estado e selecionava o best_model com base em
    reward de avaliação inválido. Aqui usamos só a métrica de rollout do treino,
    sem segundo env.

    FIX (20/06/2026): a versão anterior lia ``logger.name_to_value`` apenas em
    ``num_timesteps % check_freq == 0``, instante em que a chave costuma estar
    ausente (o SB3 só popula no dump por episódio) → ficava ``-inf`` e o
    best_model nunca era salvo. Agora a média vem de ``model.ep_info_buffer``,
    que o SB3 mantém populado com os últimos episódios encerrados.
    """

    def __init__(self, save_path: str, check_freq: int = 2_000,
                 min_episodes: int = 20, verbose: int = 1) -> None:
        super().__init__(verbose)
        self._save_path = os.path.join(save_path, "best_model.zip")
        self._check_freq = check_freq
        self._min_episodes = min_episodes
        self._best = float("-inf")

    def _on_step(self) -> bool:
        if self.num_timesteps % self._check_freq != 0:
            return True
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < self._min_episodes:
            return True
        mean_reward = sum(ep["r"] for ep in buf) / len(buf)
        if mean_reward > self._best:
            self._best = mean_reward
            self.model.save(self._save_path)
            if self.verbose:
                print(
                    f"[BestModel] novo melhor ep_rew_mean={mean_reward:.1f} "
                    f"(janela {len(buf)} eps) @ step {self.num_timesteps} "
                    f"→ {self._save_path}"
                )
        return True


class NoImprovementCallback(BaseCallback):
    """Padrão-ouro de early-stop (09/07/2026): sem melhora por N avaliações
    consecutivas, em vez de limiar fixo checado num único step (PlanBCallback,
    mais fraco — ver [[project-treino-sparse-08jul]], pesquisa sobre
    StopTrainingOnNoModelImprovement do SB3 e detecção de platô por tendência).

    Não aborta o treino sozinho — só ACUSA o platô cedo, de forma mais
    sensível que o PlanB (que só checa em planb_step). Decisão de abortar ou
    não fica com --planb-enable; este callback é só o SINAL.
    """

    def __init__(self, check_freq: int = 2_000, min_episodes: int = 20,
                 patience: int = 10, min_delta: float = 1.0,
                 verbose: int = 1) -> None:
        super().__init__(verbose)
        self._check_freq = check_freq
        self._min_episodes = min_episodes
        self._patience = patience
        self._min_delta = min_delta
        self._best = float("-inf")
        self._evals_without_improvement = 0
        self._flagged = False

    def _on_step(self) -> bool:
        if self.num_timesteps % self._check_freq != 0:
            return True
        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < self._min_episodes:
            return True
        mean_reward = sum(ep["r"] for ep in buf) / len(buf)
        if mean_reward > self._best + self._min_delta:
            self._best = mean_reward
            self._evals_without_improvement = 0
        else:
            self._evals_without_improvement += 1
        if (self._evals_without_improvement >= self._patience
                and not self._flagged):
            self._flagged = True
            print(
                f"\n[NoImprovement] PLATÔ DETECTADO — sem melhora "
                f"(Δ<{self._min_delta}) em {self._patience} avaliações "
                f"consecutivas ({self._patience * self._check_freq} steps). "
                f"Melhor ep_rew_mean={self._best:.1f} @ step {self.num_timesteps}. "
                f"Isso é um SINAL, não aborta o treino automaticamente.\n"
            )
        return True


def main() -> None:
    args = parse_args()
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    rclpy.init()
    node = _GazeboEnvNode()

    # DummyVecEnv + VecNormalize SÓ de recompensa (norm_obs=False): estabiliza o
    # crítico do SAC sem alterar a observação — assim a inferência NÃO precisa
    # das stats do VecNormalize (a política observa obs cruas).
    # info_keywords: sem isso não dá pra saber a taxa real de colisão/sucesso
    # a partir do monitor.csv/TensorBoard — só o ep_rew_mean agregado, que não
    # diferencia "quase evitou colisão" de "colidiu na largada" (ambos ~-99/-100
    # dado R_COLLISION=-100 dominante). Achado em produção 08-09/07 — ver
    # [[project-treino-sparse-08jul]].
    train_env = DummyVecEnv([
        lambda: Monitor(
            TurtleBot3GazeboEnv(node=node, seed=args.seed),
            info_keywords=("collision", "goal_reached", "curr_max_dist", "success_rate"),
        )
    ])
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True, clip_reward=10.0,
        gamma=0.99,
    )

    model_name = f"sac_{args.seed}_{args.steps}"

    best_cb = BestRolloutModelCallback(
        save_path=args.models_dir,
        check_freq=args.eval_freq,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=args.models_dir,
        name_prefix=f"sac_{args.seed}_ckpt",
        save_replay_buffer=True,   # obrigatório p/ auto-resume equivalente (SAC é off-policy)
        save_vecnormalize=True,    # preserva stats de normalização de reward entre restarts
        verbose=0,
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = args.logs_dir
    except ImportError:
        print("[WARN] tensorboard not installed — logging disabled")
        tb_log = None

    # Auto-resume: se existir checkpoint do mesmo seed, continua dele em vez de
    # treinar do zero. Isso é o que evita perder progresso quando o container
    # morre (crash, docker stop, reboot) — combinado com `restart: unless-stopped`
    # no docker-compose.yml, o treino se recupera sozinho sem intervenção manual.
    # Usar --fresh para forçar treino do zero mesmo com checkpoints existentes.
    resume_from = None if args.fresh else find_latest_checkpoint(
        args.models_dir, args.seed
    )
    # ent_coef "auto" precisa de um target_entropy explícito quando queremos
    # ser menos agressivos que o default do SB3 (-dim(ação)=-2).
    ent_coef_arg = args.ent_coef if args.ent_coef == "auto" else float(args.ent_coef)
    target_entropy_arg = (
        args.target_entropy if args.target_entropy is not None else "auto"
    )

    if resume_from is not None:
        model_path, replay_path, vecnorm_path = resume_from
        print(f"[Resume] Checkpoint encontrado: {model_path}")
        if vecnorm_path:
            train_env = VecNormalize.load(vecnorm_path, train_env.venv)
            print(f"[Resume] VecNormalize stats carregadas: {vecnorm_path}")
        # IMPORTANTE: NÃO passar ent_coef/target_entropy no load() quando o
        # checkpoint foi salvo com ent_coef FIXO — SAC.load() usa
        # set_parameters(exact_match=True) internamente, e um checkpoint fixo
        # não tem 'ent_coef_optimizer' salvo, o que causa
        # "ValueError: Names of parameters do not match" se _setup_model()
        # recriar a rede esperando esse optimizer (achado em produção, 08/07).
        # Fix: carrega NORMAL (preserva exact_match), depois troca o regime de
        # entropia manualmente SEM reconstruir a rede (preserva pesos e
        # optimizers do crítico/ator intactos).
        model = SAC.load(model_path, env=train_env, tensorboard_log=tb_log)
        print(f"[Resume] ent_coef salvo no checkpoint: {model.ent_coef}")
        if ent_coef_arg == "auto" and not isinstance(model.ent_coef, str):
            import numpy as np
            import torch as th
            model.ent_coef = "auto"
            model.target_entropy = (
                float(target_entropy_arg) if target_entropy_arg != "auto"
                else float(-np.prod(model.action_space.shape).astype(np.float32))
            )
            init_value = 1.0
            model.log_ent_coef = th.log(
                th.ones(1, device=model.device) * init_value
            ).requires_grad_(True)
            model.ent_coef_optimizer = th.optim.Adam(
                [model.log_ent_coef], lr=model.lr_schedule(1)
            )
            print(f"[Resume] Trocado para ent_coef=auto, "
                  f"target_entropy={model.target_entropy} "
                  f"(crítico/ator PRESERVADOS, só o regime de entropia mudou)")
        elif ent_coef_arg != "auto":
            model.ent_coef = ent_coef_arg
            print(f"[Resume] ent_coef mantido/definido como {ent_coef_arg}")
        if replay_path:
            model.load_replay_buffer(replay_path)
            print(f"[Resume] Replay buffer carregado: {replay_path} "
                  f"({model.replay_buffer.size()} transições)")
        else:
            print("[Resume] AVISO: replay buffer não encontrado — "
                  "retomando com buffer vazio (menos ideal, mas não do zero).")
        if args.reset_log_std_on_resume and hasattr(model.policy.actor, "log_std"):
            # log_std é um nn.Parameter direto (matriz gSDE [n_features, n_actions]
            # neste setup, não nn.Linear com weight/bias — confirmado inspecionando
            # o checkpoint real em produção, 08/07). Todos os ~512 valores estavam
            # travados perto de -3.0 (init padrão), confirmando o diagnóstico.
            import torch.nn as nn
            nn.init.constant_(model.policy.actor.log_std, 0.0)  # std≈1.0 (era ~0.05)
            print("[Resume] log_std do ator resetado (std inicial ≈1.0) — "
                  "crítico e resto do ator PRESERVADOS.")
        print(f"[Resume] Continuando de num_timesteps={model.num_timesteps}")
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=10_000,   # mundo esparso: sinal mais limpo, 10k suficiente
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,         # 1 update/passo: evita divergência de Q-values com R_APPROACH=10
            ent_coef=ent_coef_arg,
            target_entropy=target_entropy_arg,
            use_sde=True,             # gSDE: exploração suave (padrão-ouro robótica)
            sde_sample_freq=64,       # reamostra o ruído de exploração a cada 64 passos
            verbose=1,
            tensorboard_log=tb_log,
            seed=args.seed,
        )

    callbacks = [best_cb, ckpt_cb, StopOnSuccessCallback(
        threshold=args.stop_success_rate,
        patience=args.stop_patience,
        check_freq=args.eval_freq,
    ), NoImprovementCallback(
        check_freq=args.eval_freq,
        patience=5,      # 5 avaliações sem melhora (~50k steps @ eval_freq=10k default)
        min_delta=1.0,
    )]
    print(f"[StopOnSuccess] ativo: encerra se sr≥{args.stop_success_rate:.0%} "
          f"@ max-curriculum por {args.stop_patience} checagens.")
    if args.planb_enable:
        callbacks.append(PlanBCallback(
            check_step=args.planb_step,
            threshold=args.planb_threshold,
        ))
        print(f"[PlanB] habilitado: abort se ep_rew_mean<{args.planb_threshold} "
              f"@ step {args.planb_step}")
    else:
        print("[PlanB] desabilitado (default) — treino corre até o fim.")

    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        tb_log_name=f"sac_{args.seed}",
        progress_bar=False,
        # False = continua contando a partir de model.num_timesteps (resume real).
        # Se treinando do zero, num_timesteps já começa em 0 — não tem efeito.
        reset_num_timesteps=False,
    )

    final_path = os.path.join(args.models_dir, f"{model_name}_final.zip")
    model.save(final_path)
    # Salva stats do VecNormalize (recompensa). Não é necessário p/ inferência
    # (norm_obs=False), mas permite retomar o treino com normalização consistente.
    vecnorm_path = os.path.join(args.models_dir, f"{model_name}_vecnormalize.pkl")
    train_env.save(vecnorm_path)
    print(f"Saved final model: {final_path}")
    print(f"Saved VecNormalize stats: {vecnorm_path}")

    train_env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
