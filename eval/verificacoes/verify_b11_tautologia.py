"""
Verificação executável do achado B11 (docs/PLANO_CORRECAO.md).

Os "planejadores" da Fase 1 (validation_abstract/planners/rrt_star.py,
experiments_abstract/sota_comparison.py) não simulam navegação: sorteiam
sucesso de duas fórmulas lineares fechadas em função da densidade ρ.

    RRT*: success_prob = max(0.6, 0.98 - rho*1.2)   -- decresce em rho
    PPO:  success_prob = max(0.1, 0.6 + rho*0.4)    -- cresce em rho

O ρ-criterion escolhe entre as duas por um limiar em ρ. Como uma reta cai e
a outra sobe, o adaptativo é matematicamente forçado a se aproximar de
max(RRT*, PPO) -- não há navegação, colisão ou geometria envolvida.

Rodar: python3 eval/verificacoes/verify_b11_tautologia.py
"""
import numpy as np


def rrt_star(rho):
    return max(0.6, 0.98 - rho * 1.2)


def ppo(rho):
    return max(0.1, 0.6 + rho * 0.4)


def adaptive(rho, tau=0.30):
    return rrt_star(rho) if rho < tau else ppo(rho)


DENSITIES = [0.15, 0.25, 0.35, 0.45, 0.55]


def main():
    print("=== B11: tabela analítica (sem aleatoriedade) ===")
    header = f'{"rho":>6} {"RRT*":>8} {"PPO":>8} {"adapt(tau=.30)":>16} {"max(RRT*,PPO)":>16}  ótimo?'
    print(header)
    for d in DENSITIES:
        r, p, a = rrt_star(d), ppo(d), adaptive(d)
        m = max(r, p)
        otimo = "sim" if abs(a - m) < 1e-9 else "NÃO -- subótimo"
        print(f"{d:>6.2f} {r:>8.3f} {p:>8.3f} {a:>16.3f} {m:>16.3f}  {otimo}")

    media_rrt = np.mean([rrt_star(d) for d in DENSITIES])
    media_ppo = np.mean([ppo(d) for d in DENSITIES])
    media_adapt = np.mean([adaptive(d) for d in DENSITIES])
    print(f"\nMédias analíticas:  RRT*={media_rrt:.4f}  PPO={media_ppo:.4f}  adaptativo={media_adapt:.4f}")

    print("\n=== Simulação com np.random (mesmo mecanismo do mock) ===")
    np.random.seed(42)
    n = 500

    def simulate(f):
        return np.mean([np.random.random() < f(d) for d in DENSITIES for _ in range(n)])

    print(f"RRT* simulado:       {simulate(rrt_star):.4f}")
    print(f"PPO simulado:        {simulate(ppo):.4f}")
    print(f"Adaptativo simulado: {simulate(adaptive):.4f}  (relatório cita 85,3% -- não bate;")
    print("                                              padrão estrutural do B11 confirma-se")
    print("                                              de qualquer forma: ver varredura abaixo)")

    print("\n=== Varredura de tau sobre as MESMAS fórmulas (reproduz o B1) ===")
    for tau in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        m = np.mean([adaptive(d, tau) for d in DENSITIES])
        marca = "  <- mínimo" if tau == 0.20 else ""
        print(f"  tau={tau:.2f} -> média={m:.4f}{marca}")

    print("\nConclusão: o tau=0.30 citado como 'ótimo' no relatório NÃO minimiza regret")
    print("nem maximiza sucesso sobre as próprias fórmulas do mock -- tau=0.20 vence.")
    print("O cruzamento das duas retas ocorre em rho=(0.98-0.6)/(1.2+0.4)=0.2375,")
    print("não em 0.30. Ver docs/PLANO_CORRECAO.md, achado B11 e B1.")


if __name__ == "__main__":
    main()
