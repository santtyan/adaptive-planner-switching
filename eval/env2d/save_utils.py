"""
save_utils.py — backup automático antes de sobrescrever modelo treinado.

Motivação (sessão 26/07/2026, ver docs/PLANO_CORRECAO.md e
[[feedback-scripts-treino-salvam-sobrescrevendo]]): train_2d_marl.py salva
o modelo em um caminho fixo sem checkpoint intermediário. Um teste de
2.000 passos rodado só para medir tempo sobrescreveu e destruiu o único
modelo MARL de 600k passos salvo (não versionado no git, binário grande,
sem forma de recuperar). Todos os scripts train_*.py deste pacote têm o
mesmo padrão de risco: caminho de saída fixo, sem verificação de
existência prévia.

Uso: chamar safe_backup(path) IMEDIATAMENTE ANTES de qualquer
model.save(path) / torch.save(..., path) / vec_env.save(path) cujo
`path` possa já existir de um treino anterior que valha a pena preservar.
"""
import os
import shutil
import time


def safe_backup(path: str) -> str | None:
    """Se `path` já existe, renomeia para `<path>.bak-<timestamp>` antes que
    o chamador o sobrescreva. Idempotente: não falha se o arquivo não
    existir. Retorna o caminho do backup criado, ou None se não havia nada
    para preservar.

    Não faz backup automático de arquivos `.bak-*` pré-existentes (evita
    empilhar backups de backups); o chamador decide quando limpar os
    antigos.
    """
    if not os.path.exists(path):
        return None
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = f"{path}.bak-{ts}"
    shutil.copy2(path, backup_path)
    print(f"  [safe_backup] {path} já existia -- cópia preservada em {backup_path}")
    return backup_path


def safe_backup_many(*paths: str) -> list:
    """Aplica safe_backup a vários caminhos de uma vez (ex.: modelo +
    vecnormalize, que sempre são salvos juntos)."""
    return [b for p in paths if (b := safe_backup(p)) is not None]
