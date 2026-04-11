#!/bin/bash
# K-Prototypes 论文编译脚本
# 使用方法: bash compile.sh

set -e
cd "$(dirname "$0")"

# 自动添加 TeX 路径
export PATH="/usr/local/texlive/2026basic/bin/universal-darwin:$PATH"

TEX_FILE="main.tex"
OUT_NAME="main"

if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found even after PATH update"
    exit 1
fi

echo "pdflatex: $(which pdflatex)"
echo ""

echo "=== Pass 1: initial compile ==="
pdflatex -interaction=nonstopmode "$TEX_FILE" 2>&1 | tail -5

echo "=== BibTeX ==="
bibtex "$OUT_NAME" 2>&1 | tail -5

echo "=== Pass 2: resolve references ==="
pdflatex -interaction=nonstopmode "$TEX_FILE" 2>&1 | tail -5

echo "=== Pass 3: final compile ==="
pdflatex -interaction=nonstopmode "$TEX_FILE" 2>&1 | tail -5

echo ""
if [ -f "${OUT_NAME}.pdf" ]; then
    echo "SUCCESS: ${OUT_NAME}.pdf ($(du -h ${OUT_NAME}.pdf | cut -f1))"
    open "${OUT_NAME}.pdf" 2>/dev/null || true
else
    echo "FAILED: check ${OUT_NAME}.log"
fi
