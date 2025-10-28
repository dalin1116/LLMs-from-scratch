"""
GPT-2 124M 模型可视化脚本
生成模型架构图和参数分布图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_architecture_diagram():
    """创建GPT模型架构图"""
    fig, ax = plt.subplots(figsize=(14, 18))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 30)
    ax.axis('off')

    # 颜色方案
    colors = {
        'embedding': '#FFB6C1',
        'attention': '#87CEEB',
        'feedforward': '#90EE90',
        'layernorm': '#FFD700',
        'output': '#FFA07A',
        'arrow': '#666666'
    }

    y_pos = 28

    # 标题
    ax.text(5, y_pos, 'GPT-2 124M Model Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold')
    y_pos -= 1.5

    # 输入
    ax.add_patch(FancyBboxPatch((3, y_pos-0.4), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#E0E0E0', linewidth=2))
    ax.text(5, y_pos, 'Input Token IDs\n[batch, seq_len]',
            ha='center', va='center', fontsize=10, fontweight='bold')
    y_pos -= 1.5

    # Embedding层
    ax.add_patch(FancyBboxPatch((1.5, y_pos-0.5), 3, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=colors['embedding'], linewidth=2))
    ax.text(3, y_pos, 'Token Embedding\n50257 → 768\n38.6M params',
            ha='center', va='center', fontsize=9)

    ax.add_patch(FancyBboxPatch((5.5, y_pos-0.5), 3, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=colors['embedding'], linewidth=2))
    ax.text(7, y_pos, 'Position Embedding\n1024 → 768\n0.79M params',
            ha='center', va='center', fontsize=9)

    # 箭头指向相加
    ax.arrow(3, y_pos-0.7, 1.5, -0.5, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])
    ax.arrow(7, y_pos-0.7, -1.5, -0.5, head_width=0.15, head_length=0.1,
             fc=colors['arrow'], ec=colors['arrow'])

    y_pos -= 2

    # 相加节点
    ax.add_patch(plt.Circle((5, y_pos), 0.3, color='#FFF', edgecolor='black', linewidth=2))
    ax.text(5, y_pos, '+', ha='center', va='center', fontsize=16, fontweight='bold')
    y_pos -= 1

    # Dropout
    ax.add_patch(FancyBboxPatch((3.5, y_pos-0.3), 3, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#D3D3D3', linewidth=1.5))
    ax.text(5, y_pos, 'Dropout (0.1)', ha='center', va='center', fontsize=9)
    y_pos -= 1.5

    # Transformer Blocks 标题
    ax.add_patch(Rectangle((0.5, y_pos-9), 9, 9,
                           edgecolor='red', facecolor='none', linewidth=3, linestyle='--'))
    ax.text(5, y_pos+0.3, 'Transformer Block (×12)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    y_pos -= 1

    # 单个Transformer Block详细结构
    block_start_y = y_pos

    # LayerNorm 1
    ax.add_patch(FancyBboxPatch((2, y_pos-0.3), 6, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=colors['layernorm'], linewidth=1.5))
    ax.text(5, y_pos, 'LayerNorm 1 (1.5K params)',
            ha='center', va='center', fontsize=9)
    y_pos -= 1

    # Multi-Head Attention
    ax.add_patch(FancyBboxPatch((1.5, y_pos-1), 7, 2,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=colors['attention'], linewidth=2))
    ax.text(5, y_pos-0.2, 'Multi-Head Attention (12 heads)',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, y_pos-0.6, 'Q, K, V: 768 → 768 each',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-0.9, 'Head dim: 64, Out proj: 768 → 768',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-1.3, '2.36M params',
            ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 2.5

    # Dropout + Residual
    ax.add_patch(FancyBboxPatch((3, y_pos-0.3), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#D3D3D3', linewidth=1.5))
    ax.text(5, y_pos, 'Dropout + Residual Add',
            ha='center', va='center', fontsize=9)

    # 残差连接箭头
    ax.annotate('', xy=(8.5, y_pos), xytext=(8.5, block_start_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(9, y_pos + (block_start_y - y_pos)/2, 'Skip',
            ha='center', va='center', fontsize=8, color='blue', rotation=90)

    y_pos -= 1

    # LayerNorm 2
    ax.add_patch(FancyBboxPatch((2, y_pos-0.3), 6, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=colors['layernorm'], linewidth=1.5))
    ax.text(5, y_pos, 'LayerNorm 2 (1.5K params)',
            ha='center', va='center', fontsize=9)
    y_pos -= 1

    # Feed Forward Network
    ff_start_y = y_pos
    ax.add_patch(FancyBboxPatch((1.5, y_pos-1.2), 7, 2.4,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=colors['feedforward'], linewidth=2))
    ax.text(5, y_pos-0.3, 'Feed Forward Network',
            ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(5, y_pos-0.7, 'Linear1: 768 → 3072',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-1.0, 'GELU Activation',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-1.3, 'Linear2: 3072 → 768',
            ha='center', va='center', fontsize=8)
    ax.text(5, y_pos-1.7, '4.72M params',
            ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 2.8

    # Dropout + Residual
    ax.add_patch(FancyBboxPatch((3, y_pos-0.3), 4, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor='#D3D3D3', linewidth=1.5))
    ax.text(5, y_pos, 'Dropout + Residual Add',
            ha='center', va='center', fontsize=9)

    # 第二个残差连接箭头
    ax.annotate('', xy=(8.5, y_pos), xytext=(8.5, ff_start_y + 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text(9, y_pos + (ff_start_y + 0.5 - y_pos)/2, 'Skip',
            ha='center', va='center', fontsize=8, color='blue', rotation=90)

    y_pos -= 1.5

    # 单个Block总结
    ax.text(5, y_pos, 'Single Block: 7.09M params',
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    y_pos -= 1

    # 12个blocks指示
    ax.text(5, y_pos, '↓ Repeated 12 times (85.03M params) ↓',
            ha='center', va='center', fontsize=10, fontweight='bold', color='red')
    y_pos -= 1.5

    # Final LayerNorm
    ax.add_patch(FancyBboxPatch((2, y_pos-0.3), 6, 0.6,
                                boxstyle="round,pad=0.05",
                                edgecolor='black', facecolor=colors['layernorm'], linewidth=2))
    ax.text(5, y_pos, 'Final LayerNorm (1.5K params)',
            ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 1.2

    # Output Head
    ax.add_patch(FancyBboxPatch((2, y_pos-0.5), 6, 1,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=colors['output'], linewidth=2))
    ax.text(5, y_pos, 'Output Head\n768 → 50257\n38.6M params',
            ha='center', va='center', fontsize=9, fontweight='bold')
    y_pos -= 1.5

    # 输出
    ax.add_patch(FancyBboxPatch((3, y_pos-0.4), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor='#E0E0E0', linewidth=2))
    ax.text(5, y_pos, 'Output Logits\n[batch, seq_len, 50257]',
            ha='center', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('gpt_model_architecture.png', dpi=300, bbox_inches='tight')
    print("架构图已保存: gpt_model_architecture.png")


def create_parameter_distribution():
    """创建参数分布饼图和柱状图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 参数统计
    components = [
        'Token Embedding',
        'Position Embedding',
        'Transformer Blocks\n(12 blocks)',
        'Final LayerNorm',
        'Output Head'
    ]

    params = [
        38597376,   # Token Embedding
        786432,     # Position Embedding
        85026816,   # 12 Transformer Blocks
        1536,       # Final LayerNorm
        38597376    # Output Head
    ]

    params_millions = [p / 1e6 for p in params]

    # 饼图
    colors_pie = ['#FFB6C1', '#DDA0DD', '#87CEEB', '#FFD700', '#FFA07A']
    explode = (0.05, 0.05, 0.1, 0.05, 0.05)

    wedges, texts, autotexts = ax1.pie(params, labels=components, autopct='%1.1f%%',
                                         startangle=90, colors=colors_pie, explode=explode,
                                         textprops={'fontsize': 10})

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    ax1.set_title('GPT-2 124M Parameter Distribution\n(Without Weight Tying: 163M params)',
                  fontsize=14, fontweight='bold', pad=20)

    # 柱状图
    bars = ax2.barh(components, params_millions, color=colors_pie, edgecolor='black', linewidth=1.5)

    # 在柱状图上添加数值标签
    for i, (bar, val) in enumerate(zip(bars, params_millions)):
        ax2.text(val + 1, i, f'{val:.2f}M\n({params[i]:,})',
                va='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Count by Component', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(params_millions) * 1.3)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # 总参数标注
    total_params = sum(params)
    fig.text(0.5, 0.02, f'Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)',
             ha='center', fontsize=13, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig('gpt_parameter_distribution.png', dpi=300, bbox_inches='tight')
    print("参数分布图已保存: gpt_parameter_distribution.png")


def create_transformer_block_details():
    """创建单个Transformer Block的详细参数分析"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Single block components
    block_components = [
        'W_query',
        'W_key',
        'W_value',
        'out_proj',
        'LayerNorm 1',
        'FF Linear1',
        'FF Linear2',
        'LayerNorm 2'
    ]

    block_params = [
        589824,     # W_query
        589824,     # W_key
        589824,     # W_value
        590592,     # out_proj (with bias)
        1536,       # LayerNorm 1
        2362368,    # FF Linear1
        2360064,    # FF Linear2
        1536        # LayerNorm 2
    ]

    block_params_millions = [p / 1e6 for p in block_params]

    # 分组
    attention_params = sum(block_params[0:4])
    ff_params = sum(block_params[5:7])
    ln_params = sum([block_params[4], block_params[7]])

    group_labels = ['Multi-Head\nAttention', 'Feed Forward\nNetwork', 'LayerNorm\n(×2)']
    group_params = [attention_params, ff_params, ln_params]
    group_params_millions = [p / 1e6 for p in group_params]
    group_colors = ['#87CEEB', '#90EE90', '#FFD700']

    # 左图：单个Block的组件分组
    wedges, texts, autotexts = ax1.pie(group_params, labels=group_labels, autopct='%1.1f%%',
                                         startangle=90, colors=group_colors,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    ax1.set_title('Single Transformer Block\nParameter Distribution\n(7.09M params)',
                  fontsize=14, fontweight='bold', pad=20)

    # 右图：详细的参数柱状图
    colors_detailed = ['#4682B4', '#4682B4', '#4682B4', '#4682B4',
                       '#FFD700', '#228B22', '#228B22', '#FFD700']
    bars = ax2.barh(block_components, block_params_millions,
                    color=colors_detailed, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, block_params_millions)):
        ax2.text(val + 0.05, i, f'{val:.2f}M',
                va='center', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Detailed Parameter Breakdown', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, max(block_params_millions) * 1.25)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # 添加分组标注
    ax2.axhspan(-0.5, 3.5, alpha=0.15, color='#87CEEB')
    ax2.axhspan(5.5, 6.5, alpha=0.15, color='#90EE90')
    ax2.axhspan(3.5, 4.5, alpha=0.15, color='#FFD700')
    ax2.axhspan(6.5, 7.5, alpha=0.15, color='#FFD700')

    plt.tight_layout()
    plt.savefig('transformer_block_details.png', dpi=300, bbox_inches='tight')
    print("Transformer Block详细图已保存: transformer_block_details.png")


def create_comparison_chart():
    """创建Weight Tying前后的对比图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    categories = ['Token\nEmbedding', 'Position\nEmbedding', 'Transformer\nBlocks (×12)',
                  'Final\nLayerNorm', 'Output\nHead', 'TOTAL']

    without_tying = [38.60, 0.79, 85.03, 0.0015, 38.60, 163.01]
    with_tying = [38.60, 0.79, 85.03, 0.0015, 0, 124.41]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width/2, without_tying, width, label='Without Weight Tying',
                   color='#FF6B6B', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, with_tying, width, label='With Weight Tying',
                   color='#4ECDC4', edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # 只显示大于0.1M的标签
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}M',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
    ax.set_title('GPT-2 Parameter Count Comparison:\nWith vs Without Weight Tying',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加说明文本
    explanation = (
        "Weight Tying: Token Embedding和Output Head共享权重\n"
        "节省参数: 38.60M (23.7%)"
    )
    ax.text(0.98, 0.97, explanation, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('weight_tying_comparison.png', dpi=300, bbox_inches='tight')
    print("Weight Tying对比图已保存: weight_tying_comparison.png")


def create_memory_usage_chart():
    """创建内存使用情况图表"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 数据类型和对应的内存使用
    dtypes = ['float32', 'float16', 'int8']

    # Without weight tying
    without_tying_params = 163009536
    memory_without = [
        without_tying_params * 4 / (1024**2),  # float32
        without_tying_params * 2 / (1024**2),  # float16
        without_tying_params * 1 / (1024**2),  # int8
    ]

    # With weight tying
    with_tying_params = 124412160
    memory_with = [
        with_tying_params * 4 / (1024**2),  # float32
        with_tying_params * 2 / (1024**2),  # float16
        with_tying_params * 1 / (1024**2),  # int8
    ]

    x = np.arange(len(dtypes))
    width = 0.35

    # 左图：Without Weight Tying
    bars1 = ax1.bar(x, memory_without, width, color=['#FF6B6B', '#FFA07A', '#FFB6C1'],
                    edgecolor='black', linewidth=2)

    for bar, mem in zip(bars1, memory_without):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.1f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax1.set_title('Without Weight Tying\n(163M params)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dtypes, fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(memory_without) * 1.2)

    # 右图：With Weight Tying
    bars2 = ax2.bar(x, memory_with, width, color=['#4ECDC4', '#45B7D1', '#96CEB4'],
                    edgecolor='black', linewidth=2)

    for bar, mem in zip(bars2, memory_with):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mem:.1f} MB',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax2.set_title('With Weight Tying\n(124M params)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(dtypes, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, max(memory_without) * 1.2)

    plt.suptitle('GPT-2 Model Memory Requirements by Data Type',
                 fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig('memory_usage_comparison.png', dpi=300, bbox_inches='tight')
    print("内存使用对比图已保存: memory_usage_comparison.png")


if __name__ == '__main__':
    print("开始生成GPT-2模型可视化图表...")
    print("-" * 50)

    create_architecture_diagram()
    create_parameter_distribution()
    create_transformer_block_details()
    create_comparison_chart()
    create_memory_usage_chart()

    print("-" * 50)
    print("所有图表生成完成！")
    print("\n生成的文件:")
    print("1. gpt_model_architecture.png - 完整架构图")
    print("2. gpt_parameter_distribution.png - 参数分布图")
    print("3. transformer_block_details.png - Transformer Block详细分析")
    print("4. weight_tying_comparison.png - Weight Tying对比")
    print("5. memory_usage_comparison.png - 内存使用对比")
