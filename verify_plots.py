import os
from pathlib import Path

def verify_pdf_text(pdf_path):
    """检查 PDF 文件内容是否包含基本的文本特征"""
    with open(pdf_path, 'rb') as f:
        content = f.read().decode('latin-1', errors='ignore')
        # 检查常见的 Matplotlib 文本特征或字体定义
        # BT/ET 是 PDF 中的 Begin Text / End Text
        has_text = 'BT' in content and 'ET' in content
        has_font = 'Font' in content or 'Type1' in content or 'TrueType' in content
        return has_text or has_font

output_dir = Path('/Users/zhangyuxin/Desktop/improved k-prototype/paper_material')
all_good = True
found_files = list(output_dir.glob('proxy_alignment_*.pdf'))
if not found_files:
    print("No PDF files found to check!")
    all_good = False
else:
    for pdf in found_files:
        result = verify_pdf_text(pdf)
        status = 'PASS' if result else 'FAIL (No text detected)'
        print(f"Checking {pdf.name}: {status}")
        if not result: all_good = False

if all_good:
    print("ALL FIGURES VERIFIED: Text/Font data present.")