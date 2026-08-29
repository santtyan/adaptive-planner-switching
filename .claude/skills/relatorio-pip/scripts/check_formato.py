#!/usr/bin/env python3
"""Verifica os limites formais do PIP/UFG contra o estado real dos arquivos.

Uso: python3 check_formato.py
Roda a partir de qualquer lugar dentro do repositório.
"""
import re
import subprocess
import sys
from pathlib import Path

def find_repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True)
    if out.returncode != 0:
        print("ERRO: não está dentro de um repositório git.", file=sys.stderr)
        sys.exit(1)
    return Path(out.stdout.strip())

def count_resumo_chars(md_path: Path) -> int:
    """Conta os parágrafos do bloco ## Resumo, do jeito que o SIGAA conta:
    caracteres com espaço, sem contar o cabeçalho nem a nota de instrução."""
    text = md_path.read_text(encoding="utf-8")
    m = re.search(r"^## Resumo\s*\n(.*?)(?=\n## |\n---)", text, re.S | re.M)
    if not m:
        return -1
    body = m.group(1)
    # remove a linha de instrução em itálico ("*(Até 2.500 caracteres...)*")
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip() and not p.strip().startswith("*(")]
    return sum(len(p) for p in paragraphs)

def check_pdf(pdf_path: Path):
    if not pdf_path.exists():
        print(f"  PDF não encontrado em {pdf_path} (rode pdflatex antes).")
        return
    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    status = "OK" if size_mb <= 2.0 else "ESTOURA o limite de 2 MB"
    print(f"  Tamanho do PDF: {size_mb:.2f} MB  [{status}]")
    out = subprocess.run(["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True)
    if out.returncode == 0:
        pages = out.stdout.split("\f")
        total_pages = len(pages)
        corpo_pages = total_pages
        for i, p in enumerate(pages, 1):
            if "Informações Complementares" in p:
                corpo_pages = i - 1
                break
        status = "dentro do referencial" if corpo_pages <= 15 else "acima de 15 (permitido, mas confirmar necessidade)"
        print(f"  Páginas totais do PDF: {total_pages} (inclui certificados/anexos, não contam p/ limite)")
        print(f"  Páginas de corpo (até antes de Informações Complementares): {corpo_pages}  [{status}]")

def main():
    root = find_repo_root()
    md_path = root / "paper" / "relatorio_final_pip.md"
    tex_path = root / "paper" / "relatorio_final_pip.tex"
    pdf_path = root / "paper" / "relatorio_final_pip.pdf"

    print("=== Resumo (limite: 2.500 caracteres com espaço) ===")
    chars = count_resumo_chars(md_path)
    if chars < 0:
        print(f"  Não encontrei o bloco '## Resumo' em {md_path}")
    else:
        status = "OK" if chars <= 2500 else f"ESTOURA em {chars - 2500} caracteres"
        print(f"  {md_path.name}: {chars} chars  [{status}]")

    print("\n=== PDF (limite: 2 MB; páginas: 15 referencial) ===")
    check_pdf(pdf_path)

    print("\n=== Divergência .md vs .tex (o .tex é o que compila para o PDF) ===")
    if md_path.exists() and tex_path.exists():
        md_mtime = md_path.stat().st_mtime
        tex_mtime = tex_path.stat().st_mtime
        if md_mtime > tex_mtime:
            print(f"  ⚠️  {md_path.name} foi modificado DEPOIS de {tex_path.name}.")
            print("      Confirme se a correção foi aplicada nos dois arquivos antes de compilar.")
        else:
            print(f"  {tex_path.name} não está desatualizado em relação ao .md por mtime "
                  "(não garante conteúdo idêntico, só que o .tex foi tocado depois).")

    print("\n=== Certificados Diálogos em Pesquisa e Inovação (OBRIGATÓRIO) ===")
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    tex_text = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    placeholder_md = "[Anexar certificados" in text
    tem_includepdf = "\\includepdf" in tex_text
    cert_dir = tex_path.parent / "certificados"
    arquivos_cert = list(cert_dir.glob("*.pdf")) if cert_dir.exists() else []
    if placeholder_md:
        print("  ⚠️  .md ainda tem marcador de placeholder ([Anexar certificados...]) — "
              "substituir pela lista real antes de considerar pronto.")
    elif tem_includepdf and arquivos_cert:
        print(f"  OK: {len(arquivos_cert)} PDF(s) de certificado em {cert_dir.name}/, "
              f"referenciados via \\includepdf no .tex.")
    elif tem_includepdf and not arquivos_cert:
        print(f"  ⚠️  .tex usa \\includepdf mas {cert_dir.name}/ não existe ou está vazia — "
              "a compilação vai falhar ou os certificados não estarão no PDF final.")
    else:
        print("  ⚠️  Nenhum \\includepdf encontrado no .tex — certificados podem estar "
              "só mencionados em texto, sem PDF de fato anexado ao documento final.")

if __name__ == "__main__":
    main()
