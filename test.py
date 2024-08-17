import torch
import torch.nn as nn
import numpy as np
import sys
from tqdm import tqdm
import torch.optim as optim
from utils.plot import plot_list_all
from utils.utils import (
    load_model,
    save_as_zip,
    clear_files,
    delete_files,
    save_checkpoint,
    get_loaders,
    get_seg_acc,
    get_dice_loss,
    get_confusion_matrix_index,
)

# 示例二值化分割结果
pred = np.array([
    [0, 1, 1],
    [0, 1, 0],
    [0, 0, 0]
])

true = np.array([
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
])

tp = 0
fp = 0
fn = 0
tn = 0

# 计算混淆就矩阵4项指标 TP,FP,FN,TN
TP, FP, FN, TN = get_confusion_matrix_index(real=true, pred=pred)

# 累加指标
tp += TP
fp += FP
fn += FN
tn += TN

# 计算指标
dice = (2 * tp) / (fn + 2 * tp + fp + 1e-8)
rec = tp / (tp + fn + 1e-8)
pre = tp / (tp + fp + 1e-8)
fpr = fp / (fp + tn + 1e-8)  # 假阳性率
fnr = fn / (tp + fn + 1e-8)  # 假阴性率

seg_result_dict = {"dice": dice, "rec": rec, "pre": pre, "fpr": fpr, "fnr": fnr}

print(seg_result_dict)
