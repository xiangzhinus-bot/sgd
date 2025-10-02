# plot_history.py
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_from_csv(file_path):
    """
    从CSV文件中读取训练历史并生成图表。
    """
    try:
        # 1. 使用pandas读取CSV文件
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{file_path}'。请检查文件名或路径是否正确。")
        return

    # 2. 准备绘图数据
    epochs = df['epoch']
    losses = df['loss']
    test_rmses = df['test_rmse']

    # 3. 绘制双Y轴图表 (与之前逻辑完全相同)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练损失
    color = 'tab:red'
    ax1.set_xlabel('Epoch (k)')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, losses, color=color, marker='o', linestyle='-', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建共享X轴的第二个Y轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Test RMSE', color=color)
    ax2.plot(epochs, test_rmses, color=color, marker='x', linestyle='--', label='Test RMSE')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和网格
    ax1.set_title('Training History')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()  # 调整布局以防止标签重叠

    # 4. 保存图表
    output_filename = file_path.replace('.csv', '.png')
    plt.savefig(output_filename)
    print(f"图表已保存为 '{output_filename}'")
    # plt.show() # 如果你想在本地直接显示图表，可以取消这行注释

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从CSV文件绘制训练历史图。")
    parser.add_argument("--file", type=str, required=True, help="包含训练历史的CSV文件路径。")
    args = parser.parse_args()
    
    plot_from_csv(args.file)
