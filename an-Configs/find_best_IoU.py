# -*- coding: utf-8 -*-
"""
提取 mmseg 日志中 tongue 的 IoU / Dice，
输出 IoU Top-3、Dice 最优 epoch，
并在无显示环境下保存曲线图。
"""
import re
import os
import matplotlib
matplotlib.use('Agg')  # ← 关键：使用非交互后端
import matplotlib.pyplot as plt

log_path = '/Users/anxingle/workspace/medical/mmsegmentation/an-Configs/train.log'  # ← 修改为你的日志文件路径

# 用正则匹配 epoch 号 和 tongue IoU/Dice 值
# === 正则：Epoch 与 tongue 行 ===
epoch_pat = re.compile(r'Epoch\(val\)\s+\[(\d+)\]')
# 形如：|   tongue   | 95.53 | 97.63 | 97.71 |
tongue_row_pat = re.compile(
    r'\|\s*tongue\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|'
)

if not os.path.isfile(log_path):
    raise FileNotFoundError(f'日志文件不存在：{log_path}')

epoch_to_metrics = {}
current_epoch = None

# === 解析日志 ===
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        m_e = epoch_pat.search(line)
        if m_e:
            current_epoch = int(m_e.group(1))
            continue
        m_t = tongue_row_pat.search(line)
        if m_t and current_epoch is not None:
            iou = float(m_t.group(1))
            acc = float(m_t.group(2))
            dice = float(m_t.group(3))
            # 同一 epoch 若多次出现，保留最后一次
            epoch_to_metrics[current_epoch] = {'iou': iou, 'dice': dice, 'acc': acc}

if not epoch_to_metrics:
    raise RuntimeError('未在日志中解析到任何 tongue 指标，请检查日志格式。')

# === IoU Top-3 ===
sorted_by_iou = sorted(epoch_to_metrics.items(), key=lambda kv: kv[1]['iou'], reverse=True)
top3_iou = sorted_by_iou[:3]

print('\n==================  Top-3 Epochs by Tongue IoU  ==================')
for rank, (ep, m) in enumerate(top3_iou, 1):
    print(f'#{rank}: epoch={ep:>4d}   IoU={m["iou"]:.4f}   Dice={m["dice"]:.4f}   Acc={m["acc"]:.4f}')

# === Dice Top-3（新增） ===
sorted_by_dice = sorted(epoch_to_metrics.items(), key=lambda kv: kv[1]['dice'], reverse=True)
top3_dice = sorted_by_dice[:3]

print('\n==================  Top-3 Epochs by Tongue Dice  ==================')
for rank, (ep, m) in enumerate(top3_dice, 1):
    print(f'#{rank}: epoch={ep:>4d}   Dice={m["dice"]:.4f}   IoU={m["iou"]:.4f}   Acc={m["acc"]:.4f}')

# === Dice 最优 ===
best_dice_ep, best_dice_metrics = sorted_by_dice[0]
print('\n==================  Best Epoch by Tongue Dice  ===================')
print(f'Best Dice Epoch: {best_dice_ep}   Dice={best_dice_metrics["dice"]:.4f}   '
      f'IoU={best_dice_metrics["iou"]:.4f}   Acc={best_dice_metrics["acc"]:.4f}')

# === 画图并保存 ===
epochs = sorted(epoch_to_metrics.keys())
ious = [epoch_to_metrics[e]['iou'] for e in epochs]
dices = [epoch_to_metrics[e]['dice'] for e in epochs]

plt.figure()
plt.plot(epochs, ious, marker='o')
plt.title('Tongue IoU vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('IoU (%)')
plt.grid(True)
plt.tight_layout()
iou_png = 'tongue_iou_vs_epoch.png'
plt.savefig(iou_png, dpi=150)

plt.figure()
plt.plot(epochs, dices, marker='o')
plt.title('Tongue Dice vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Dice (%)')
plt.grid(True)
plt.tight_layout()
dice_png = 'tongue_dice_vs_epoch.png'
plt.savefig(dice_png, dpi=150)

print(f'\n图已保存：\n - {os.path.abspath(iou_png)}\n - {os.path.abspath(dice_png)}')
print('Done.')