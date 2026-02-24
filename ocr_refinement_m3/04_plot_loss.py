import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac通用中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 1. 解析日志数据
logs = [
    {'loss': 2.6295, 'grad_norm': 1.2176889181137085, 'learning_rate': 0.00011676923076923076, 'epoch': 1.83},
    {'loss': 3.201, 'grad_norm': 1.2467368841171265, 'learning_rate': 0.00011646153846153845, 'epoch': 1.84},
    {'loss': 2.401, 'grad_norm': 1.7756755352020264, 'learning_rate': 0.00011615384615384615, 'epoch': 1.84},
    {'loss': 2.5141, 'grad_norm': 1.238408088684082, 'learning_rate': 0.00011584615384615385, 'epoch': 1.84},
    {'loss': 2.3513, 'grad_norm': 2.176553964614868, 'learning_rate': 0.00011553846153846152, 'epoch': 1.85},
    {'loss': 2.1561, 'grad_norm': 1.5579423904418945, 'learning_rate': 0.00011523076923076922, 'epoch': 1.85},
    {'loss': 0.5873, 'grad_norm': 0.9161090850830078, 'learning_rate': 0.00011492307692307691, 'epoch': 1.85},
    {'loss': 0.944, 'grad_norm': 0.7956250309944153, 'learning_rate': 0.0001146153846153846, 'epoch': 1.86},
    {'loss': 3.4488, 'grad_norm': 0.37122973799705505, 'learning_rate': 0.0001143076923076923, 'epoch': 1.86},
    {'loss': 1.8793, 'grad_norm': 1.1950807571411133, 'learning_rate': 0.00011399999999999999, 'epoch': 1.86},
    {'loss': 2.1456, 'grad_norm': 1.2985639572143555, 'learning_rate': 0.00011369230769230769, 'epoch': 1.86},
    {'loss': 2.8941, 'grad_norm': 1.01361882686615, 'learning_rate': 0.00011338461538461537, 'epoch': 1.87},
    {'loss': 2.1562, 'grad_norm': 0.9296738505363464, 'learning_rate': 0.00011307692307692307, 'epoch': 1.87},
    {'loss': 2.114, 'grad_norm': 1.1550109386444092, 'learning_rate': 0.00011276923076923075, 'epoch': 1.87},
    {'loss': 2.7493, 'grad_norm': 1.589343786239624, 'learning_rate': 0.00011246153846153845, 'epoch': 1.88},
    {'loss': 2.7976, 'grad_norm': 1.2335491180419922, 'learning_rate': 0.00011215384615384614, 'epoch': 1.88},
    {'loss': 2.0409, 'grad_norm': 1.0995981693267822, 'learning_rate': 0.00011184615384615383, 'epoch': 1.88},
    {'loss': 2.1017, 'grad_norm': 1.4522318840026855, 'learning_rate': 0.00011153846153846153, 'epoch': 1.89},
    {'loss': 2.6674, 'grad_norm': 1.3295310735702515, 'learning_rate': 0.00011123076923076923, 'epoch': 1.89},
    {'loss': 3.7155, 'grad_norm': 1.174507975578308, 'learning_rate': 0.00011092307692307691, 'epoch': 1.89},
    {'loss': 2.961, 'grad_norm': 1.2085458040237427, 'learning_rate': 0.00011061538461538461, 'epoch': 1.9},
    {'loss': 3.5525, 'grad_norm': 1.3515263795852661, 'learning_rate': 0.0001103076923076923, 'epoch': 1.9},
    {'loss': 1.8047, 'grad_norm': 1.0266848802566528, 'learning_rate': 0.00010999999999999998, 'epoch': 1.9},
    {'loss': 3.0086, 'grad_norm': 1.1100993156433105, 'learning_rate': 0.00010969230769230767, 'epoch': 1.9},
    {'loss': 0.6065, 'grad_norm': 0.9431750178337097, 'learning_rate': 0.00010938461538461537, 'epoch': 1.91},
    {'loss': 1.3926, 'grad_norm': 0.946526288986206, 'learning_rate': 0.00010907692307692307, 'epoch': 1.91},
    {'loss': 2.6703, 'grad_norm': 0.9859640002250671, 'learning_rate': 0.00010876923076923075, 'epoch': 1.91},
    {'loss': 1.8874, 'grad_norm': 1.1354775428771973, 'learning_rate': 0.00010846153846153845, 'epoch': 1.92},
    {'loss': 2.6429, 'grad_norm': 1.4037634134292603, 'learning_rate': 0.00010815384615384615, 'epoch': 1.92},
    {'loss': 3.6908, 'grad_norm': 1.224959135055542, 'learning_rate': 0.00010784615384615384, 'epoch': 1.92},
    {'loss': 2.5813, 'grad_norm': 1.098358154296875, 'learning_rate': 0.00010753846153846153, 'epoch': 1.93}
]

# 2. 提取epoch和loss数据
epochs = [log['epoch'] for log in logs]
losses = [log['loss'] for log in logs]

# 3. 计算滑动平均（平滑曲线，更易观察趋势）
window_size = 3
smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
smoothed_epochs = epochs[window_size-1:]

# 4. 可视化loss变化趋势
plt.figure(figsize=(12, 6))
# 原始loss曲线
plt.plot(epochs, losses, 'b-', alpha=0.5, label='原始Loss', markersize=4)
# 平滑后的loss曲线
plt.plot(smoothed_epochs, smoothed_losses, 'r-', linewidth=2, label=f'滑动平均Loss (窗口={window_size})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('模型训练Loss变化趋势')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 5. 数值分析：计算关键指标
print("==== Loss 统计分析 ====")
print(f"Loss 最大值: {np.max(losses):.4f}")
print(f"Loss 最小值: {np.min(losses):.4f}")
print(f"Loss 平均值: {np.mean(losses):.4f}")
print(f"Loss 标准差: {np.std(losses):.4f}")

# 分析趋势：计算最后10个loss的均值 vs 前10个loss的均值
if len(losses) >= 10:
    first_10_mean = np.mean(losses[:10])
    last_10_mean = np.mean(losses[-10:])
    print(f"\n前10个Loss均值: {first_10_mean:.4f}")
    print(f"最后10个Loss均值: {last_10_mean:.4f}")
    print(f"变化率: {(last_10_mean - first_10_mean)/first_10_mean * 100:.2f}%")