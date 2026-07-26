"""
diagnose_physics_window.py — Mede, sem treinar nada, o que acontece dentro
do ciclo real de step() (pause -> publish -> unpause -> wait_for_scan ->
pause) usado por gazebo_gym_env.py.

Objetivo: confirmar ou refutar a hipótese registrada (max_wheel_acceleration
não tem tempo de agir dentro da janela de física despausada) medindo
diretamente, em vez de inferir por comparação qualitativa como o diagnóstico
anterior (DEVELOPMENT_LOG.md, achado #7) fez.

Para cada um de N passos com comando constante (v=LINEAR_VEL_MAX, omega=0):
  - tempo de relógio REAL decorrido entre unpause_physics() e o próximo
    pause_physics() (ou seja, a janela de física realmente despausada)
  - deslocamento (x,y) do robô nesse passo
  - velocidade linear resultante (deslocamento / DT nominal do controle)

Compara com um bloco de M segundos de comando constante SEM pausar a
física entre publicações (o cenário "comando manual sustentado" do
diagnóstico anterior) — mesma métrica, para ficar lado a lado.

Uso (dentro do container, com Gazebo já rodando e o waffle já spawnado
num mundo qualquer):
    python3 diagnose_physics_window.py
"""
import time
import sys

import rclpy

sys.path.insert(0, "/workspace/ros2_ws/src/turtlebot3_gym_env")
from turtlebot3_gym_env.gazebo_gym_env import (
    _GazeboEnvNode, LINEAR_VEL_MAX, SCAN_TIMEOUT,
)

N_STEPS_PIPELINE = 15
T_SUSTAINED_SECONDS = 3.0


def get_pose(node):
    return node.get_robot_pose()


def main():
    rclpy.init()
    node = _GazeboEnvNode()

    # Aguarda scan e pose iniciais chegarem.
    t0 = time.time()
    while node._scan is None and time.time() - t0 < 10.0:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node._scan is None:
        print("ERRO: nenhum /scan recebido em 10s -- Gazebo/waffle não está pronto.")
        return

    # CRÍTICO: get_robot_pose() só sai do fallback _cached_pose (fixo em
    # (0,0,0)) depois que teleport_robot() ancora _odom_ref/_world_ref (ver
    # gazebo_gym_env.py). Sem isso, toda medição de "deslocamento" mede o
    # cache estático, não a física real -- foi o que aconteceu na 1ª rodada
    # deste diagnóstico e produziu falso positivo (0.00mm em tudo).
    print("Teleportando o robô para ancorar _odom_ref/_world_ref (como reset() faz)...")
    node.pause_physics(blocking=False)
    node.teleport_robot(-2.0, -0.5, 0.0)
    node.unpause_physics(blocking=False)
    node.wait_for_scan(timeout=2.0)
    node.pause_physics(blocking=False)
    print(f"Pose pós-teleport: {get_pose(node)}\n")

    print("=== Diagnóstico: ciclo real do step() (pause/publish/unpause/wait_scan/pause) ===\n")

    windows_ms = []
    displacements = []
    x0, y0, _ = get_pose(node)
    for i in range(N_STEPS_PIPELINE):
        xa, ya, _ = get_pose(node)

        node.pause_physics(blocking=False)
        node.publish_cmd(LINEAR_VEL_MAX, 0.0)
        t_unpause = time.time()
        node.unpause_physics(blocking=False)

        scan = node.wait_for_scan(timeout=SCAN_TIMEOUT)
        t_pause = time.time()
        node.pause_physics(blocking=False)

        window_ms = (t_pause - t_unpause) * 1000.0
        xb, yb, _ = get_pose(node)
        disp = ((xb - xa) ** 2 + (yb - ya) ** 2) ** 0.5

        windows_ms.append(window_ms)
        displacements.append(disp)
        print(f"  passo {i:2d}: janela despausada = {window_ms:7.1f} ms   "
              f"deslocamento = {disp*1000:6.2f} mm")

    xf, yf, _ = get_pose(node)
    total_disp_pipeline = ((xf - x0) ** 2 + (yf - y0) ** 2) ** 0.5
    avg_window = sum(windows_ms) / len(windows_ms)
    avg_disp = sum(displacements) / len(displacements)

    print(f"\n  Janela média despausada: {avg_window:.1f} ms")
    print(f"  Deslocamento médio/passo: {avg_disp*1000:.2f} mm")
    print(f"  Deslocamento TOTAL em {N_STEPS_PIPELINE} passos: {total_disp_pipeline:.3f} m")
    print(f"  Velocidade efetiva média: {total_disp_pipeline / (N_STEPS_PIPELINE * avg_window / 1000):.4f} m/s "
          f"(comandado: {LINEAR_VEL_MAX:.4f} m/s)")

    print(f"\n=== Comparação: comando constante SEM pausar física, por {T_SUSTAINED_SECONDS:.1f}s ===\n")
    node.unpause_physics(blocking=False)
    xs0, ys0, _ = get_pose(node)
    t_start = time.time()
    while time.time() - t_start < T_SUSTAINED_SECONDS:
        node.publish_cmd(LINEAR_VEL_MAX, 0.0)
        rclpy.spin_once(node, timeout_sec=0.05)
    xs1, ys1, _ = get_pose(node)
    node.publish_cmd(0.0, 0.0)
    node.pause_physics(blocking=False)

    disp_sustained = ((xs1 - xs0) ** 2 + (ys1 - ys0) ** 2) ** 0.5
    v_sustained = disp_sustained / T_SUSTAINED_SECONDS
    print(f"  Deslocamento em {T_SUSTAINED_SECONDS:.1f}s sustentado: {disp_sustained:.3f} m")
    print(f"  Velocidade efetiva: {v_sustained:.4f} m/s (comandado: {LINEAR_VEL_MAX:.4f} m/s)")

    print("\n=== CONCLUSÃO ===")
    ratio = (total_disp_pipeline / (N_STEPS_PIPELINE * avg_window / 1000)) / max(v_sustained, 1e-6)
    print(f"  Razão velocidade-efetiva(pipeline) / velocidade-efetiva(sustentado): {ratio:.3f}")
    if ratio < 0.3:
        print("  >>> CONFIRMA a hipótese: a janela de física despausada é curta demais para o "
              "robô atingir velocidade comparável ao comando sustentado.")
    elif avg_window < 50:
        print(f"  >>> Janela média de {avg_window:.1f} ms é MUITO curta -- possível causa raiz "
              "é o wait_for_scan retornando quase instantaneamente (scan já pronto/stale), "
              "não a aceleração das rodas em si. Investigar SCAN_TIMEOUT vs. update_rate do LIDAR.")
    else:
        print("  >>> NÃO confirma a hipótese de aceleração insuficiente -- deslocamento no "
              "pipeline é comparável ao sustentado. Causa raiz provavelmente é outra "
              "(verificar se pause_physics/unpause_physics estão de fato surtindo efeito, "
              "ou se get_robot_pose() está lendo uma pose desatualizada).")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
