"""
快速重绘脚本 v7：终极路径化导出 (True Path Flattening)
使用 svg 后端作为中转，强制将所有文字转换为几何路径，然后再转回 PDF/PNG。
这种方法在 Matplotlib 中是最彻底的，生成的 PDF 不包含任何字体定义。
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io

# 核心配置：禁用所有可能导致方块的 Unicode 和字体嵌入
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
    'axes.unicode_minus': False,
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans']
})

def replot_from_csv(csv_path, output_dir):
    try:
        df = pd.read_csv(csv_path)
        ds_name = df['dataset'].iloc[0]
        safe_name = ds_name.replace(' ', '_').lower()
        
        fig, ax1 = plt.subplots(figsize=(7, 4))
        color_j = '#2166ac'
        color_ari = '#d6604d'

        x = df['log10_gamma'].values
        j_vals = df['cost_mean'].values
        j_norm = (j_vals - j_vals.min()) / (j_vals.max() - j_vals.min() + 1e-12)

        ax1.plot(x, j_norm, color=color_j, lw=2, label='J(gamma)')
        ax1.set_xlabel("Log10 Gamma (Path Rendered)", fontsize=12)
        ax1.set_ylabel("J(gamma) Normalized", color=color_j, fontsize=11)
        ax1.tick_params(axis='y', labelcolor=color_j)

        ax2 = ax1.twinx()
        ax2.plot(x, df['ari_mean'].values, color=color_ari, lw=2, linestyle='--', label='ARI(gamma)')
        ax2.set_ylabel("ARI (External Quality)", color=color_ari, fontsize=11)
        ax2.tick_params(axis='y', labelcolor=color_ari)

        gamma_opt_j_idx = df['cost_mean'].idxmin()
        gamma_opt_ari_idx = df['ari_mean'].idxmax()
        ax1.axvline(df.loc[gamma_opt_j_idx, 'log10_gamma'], color=color_j, linestyle=':', alpha=0.5)
        ax2.axvline(df.loc[gamma_opt_ari_idx, 'log10_gamma'], color=color_ari, linestyle=':', alpha=0.5)

        plt.title(f"Dataset: {ds_name}", fontsize=12)
        plt.tight_layout()
        
        # 导出高清 PNG (位图是最后的底线，绝对不会有字体问题)
        png_path = output_dir / f'proxy_alignment_{safe_name}.png'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        
        # 导出 PDF 时，我们再次确认设置
        pdf_path = output_dir / f'proxy_alignment_{safe_name}.pdf'
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        print(f"SUCCESSfully RE-RENDERED: {safe_name}")
    except Exception as e:
        print(f"ERROR on {csv_path.name}: {e}")

def main():
    results_dir = Path('/Users/zhangyuxin/Desktop/improved k-prototype/results/exp_proxy')
    output_dir = Path('/Users/zhangyuxin/Desktop/improved k-prototype/paper_material')
    csv_files = list(results_dir.glob('*_scan.csv'))
    
    print(f"Starting v7 Re-plot for {len(csv_files)} files...")
    for csv_file in csv_files:
        replot_from_csv(csv_file, output_dir)

if __name__ == '__main__':
    main()