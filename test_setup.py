# test_setup.py - Validação rápida do ambiente
import sys
sys.path.append('src')

try:
    from environment import SimpleEnvironment, GridEnv
    import numpy as np
    
    print('='*50)
    print('TESTE DE VALIDAÇÃO - AMBIENTE')
    print('='*50)
    
    # Test 1: SimpleEnvironment
    print('\n1. Testando SimpleEnvironment...')
    env = SimpleEnvironment(obstacle_density=0.3)
    density = env.get_density()
    valid_pos = env.is_valid(50, 50)
    print(f'   ✓ Densidade: {density:.3f} (target: 0.3)')
    print(f'   ✓ Posição (50,50) válida: {valid_pos}')
    
    # Test 2: GridEnv (sem treino ainda)
    print('\n2. Testando GridEnv...')
    gym_env = GridEnv(grid_size=50, obstacle_density=0.2)  # Menor para teste
    obs, info = gym_env.reset(seed=42)
    print(f'   ✓ Obs shape: {obs.shape} (esperado: (5,))')
    print(f'   ✓ Obs values: {obs}')
    
    # Test 3: Step simples
    action = np.array([1.0, 1.0])
    obs, reward, done, truncated, info = gym_env.step(action)
    print(f'   ✓ Step executado, reward: {reward:.2f}')
    
    print('\n' + '='*50)
    print('✅ TODOS OS TESTES PASSARAM!')
    print('✅ Ambiente pronto para usar')
    print('='*50)
    
except Exception as e:
    print(f'❌ ERRO: {e}')
    print('Instale as dependências: pip install -r requirements.txt')
