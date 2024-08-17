"""
训练脚本

目的：
- 验证分类引导思路CGM有效性

注意：
- 默认unet构型的最深层通道数为512；
- 假阳性率 fpr 在保存入list变量前，会先扩大1e4倍，以使得值在0~1之间；
"""

import torch
import torch.nn as nn
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
)

# from model.unet3d.model import Generator
from model.encoder_class.model import Generator  # 分类网络路径

# 超参数
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
isLinux = False
NUM_EPOCHS = 1000
TEST_INDEXES_PER_EPOCH = 5  # 计算指标间隔

# 数据集路径
TRAIN_DIR = r"data/train"
TEST_DIR = r"data/test"

# 日志文件路径
TRAIN_INDEXES_DIR = r"log/train_indexes.txt"  # indexes in train
TEST_INDEXES_DIR = r"log/test_indexes.txt"  # indexes in test
TRAIN_LOSS_G_DIR = r"log/train_loss_g.txt"  # train loss

# Model
generator = Generator().to(DEVICE)
# 加载已有模型
# model_path = "checkpoint/my_checkpoint_epoch_400.pth.tar"
# load_model(checkpoint=torch.load(model_path), model=generator, model_state_dict_name="state_dict_G")

optimizerG = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

train_loader = get_loaders(data_dir=TRAIN_DIR,
                           batch_size=BATCH_SIZE,
                           isLinux=isLinux,
                           isShuffle=True)
test_loader = get_loaders(data_dir=TEST_DIR,
                          batch_size=BATCH_SIZE,
                          isLinux=isLinux,
                          isShuffle=False)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Train <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
loss_g_item = 0.0
loss_g_list = []

# Train indexes
train_dice_list = []
train_rec_list = []
train_pre_list = []
train_fpr_list = []
train_fnr_list = []

# Test indexes
test_dice_list = []
test_pre_list = []
test_rec_list = []
test_fpr_list = []
test_fnr_list = []

# Clear record
clear_files(directory='./log/')  # logs
delete_files(directory='./save/')  # checkpoints

generator.train()

for epoch in tqdm(range(NUM_EPOCHS)):  # 0 ~ (NUM_EPOCHS-1)
    for i, data in enumerate(train_loader):
        # 添加通道维度: [N, 112, 112, 112] -> [N, 1, 112, 112, 112]
        images = data[0].unsqueeze(1).to(DEVICE)  # image
        gt = data[1].unsqueeze(1).to(DEVICE)  # gt

        generator.zero_grad()

        seg_out, pred = generator(images)
        seg_out = torch.sigmoid(seg_out)  # 分割网络预测图
        pred = torch.sigmoid(pred)  # 整体预测图

        # pred = torch.sigmoid(generator(images))  # [N, 1, 112, 112, 112]

        loss_G = get_dice_loss(seg_out, gt) + get_dice_loss(pred, gt)  # Dice损失
        # loss_g_bce = nn.BCEWithLogitsLoss()(generator(images), mask_real)  # BCE损失

        loss_G.backward()
        optimizerG.step()

        loss_g_item = loss_G.item()

        # 一个iteration结束
        print(
            f"Epoch_{epoch}_i_{i}:  loss_g_dice:{loss_g_item:.6f}"
        )

    # 一个epoch结束
    loss_g_list.append(loss_g_item)

    # 每隔指定间隔计算训练集、测试集指标，保存loss，绘制训练参数曲线图
    if (epoch + 1) % TEST_INDEXES_PER_EPOCH == 0:
        # ------------------------------ 训练集指标 ------------------------------
        train_result_dict = get_seg_acc(train_loader, generator, device=DEVICE)
        train_dice_list.append(train_result_dict["dice"])
        train_pre_list.append(train_result_dict["pre"])
        train_rec_list.append(train_result_dict["rec"])
        train_fpr_list.append(train_result_dict["fpr"])
        train_fnr_list.append(train_result_dict["fnr"])
        print(
            f"Train_Epoch_{epoch + 1}: ------ dice:{train_dice_list[-1]:.6f}    pre:{train_pre_list[-1]:.6f}    rec:{train_rec_list[-1]:.6f}    fpr:{train_fpr_list[-1]:.6f}    fnr:{train_fnr_list[-1]:.6f}\n"
        )

        # 写入日志
        f = open(TRAIN_INDEXES_DIR, "a")
        f.write(
            f"Epoch_{epoch + 1}: --- dice:{train_dice_list[-1]:.6f}    pre:{train_pre_list[-1]:.6f}    rec:{train_rec_list[-1]:.6f}    fpr:{train_fpr_list[-1]:.6f}    fnr:{train_fnr_list[-1]:.6f}\n"
        )
        f.close()

        # ------------------------------ 测试集指标 ------------------------------
        test_result_dict = get_seg_acc(test_loader, generator, device=DEVICE)
        test_dice_list.append(test_result_dict["dice"])
        test_pre_list.append(test_result_dict["pre"])
        test_rec_list.append(test_result_dict["rec"])
        test_fpr_list.append(test_result_dict["fpr"])
        test_fnr_list.append(test_result_dict["fnr"])
        print(
            f"Test_Epoch_{epoch + 1}: --- dice:{test_dice_list[-1]:.6f}  pre:{test_pre_list[-1]:.6f}  rec:{test_rec_list[-1]:.6f}    fpr:{test_fpr_list[-1]:.6f}    fnr:{test_fnr_list[-1]:.6f}\n"
        )

        # 写入日志
        f = open(TEST_INDEXES_DIR, "a")
        f.write(
            f"Epoch_{epoch + 1}: --- dice:{test_dice_list[-1]:.6f}  pre:{test_pre_list[-1]:.6f}  rec:{test_rec_list[-1]:.6f}    fpr:{test_fpr_list[-1]:.6f}    fnr:{test_fnr_list[-1]:.6f}\n"
        )
        f.close()

        # ------------------------------ 保存分割损失 ------------------------------
        # 写入日志
        f = open(TRAIN_LOSS_G_DIR, "a")  # G loss
        f.write(
            f"Epoch_{epoch + 1}:  loss_g:{loss_g_list[-1]:.6f}\n")
        f.close()

        # ------------------------------ 绘制参数曲线图 ------------------------------
        # 训练集
        plot_list_all(all_list=[train_dice_list, train_pre_list, train_rec_list, train_fpr_list, train_fnr_list],
                      all_list_name=['Dice', 'Pre', 'Rec', 'Fpr', 'Fnr'],
                      save_path='save/train_plot/',
                      file_name='train_indexes_list',
                      currentEpoch=epoch,
                      step=TEST_INDEXES_PER_EPOCH)
        # 损失
        plot_list_all(all_list=[loss_g_list],
                      all_list_name=['Dice loss'],
                      save_path='save/train_plot/',
                      file_name='loss_g_list',
                      currentEpoch=epoch,
                      step=TEST_INDEXES_PER_EPOCH)
        # 测试集
        plot_list_all(all_list=[test_dice_list, test_pre_list, test_rec_list, test_fpr_list, test_fnr_list],
                      all_list_name=['Dice', 'Pre', 'Rec', 'Fpr', 'Fnr'],
                      save_path='save/train_plot/',
                      file_name='test_indexes_list',
                      currentEpoch=epoch,
                      step=TEST_INDEXES_PER_EPOCH)

    # 保存模型参数、关键数据
    if (epoch + 1) % TEST_INDEXES_PER_EPOCH == 0:
        # 字典
        checkpoint = {
            # 模型参数
            "state_dict_G": generator.state_dict(),  # 生成器模型参数

            # 损失函数
            "loss_g_list": loss_g_list,

            # 当前epoch
            "currentEpoch": epoch + 1,

            # 模型在训练集的分割指标
            "train_dice_list": train_dice_list,
            "train_rec_list": train_rec_list,
            "train_pre_list": train_pre_list,
            "train_fpr_list": train_fpr_list,
            "train_fnr_list": train_fnr_list,

            # 模型在测试集的分割指标
            "test_dice_list": test_dice_list,
            "test_pre_list": test_pre_list,
            "test_rec_list": test_rec_list,
            "test_fpr_list": test_fpr_list,
            "test_fnr_list": test_fnr_list,
        }
        save_checkpoint(checkpoint,
                        filename="checkpoint/my_checkpoint_epoch_" + str(epoch + 1) + ".pth.tar")

# ------------------------------ 测试结束，保存结果 ------------------------------
save_as_zip(target_path="log", output_path="log.zip")
save_as_zip(target_path="save", output_path="save.zip")
