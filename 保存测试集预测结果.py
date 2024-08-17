import torch
import sys
import os
import nibabel as nib
from utils.files import create_directory_if_not_exists
from utils.utils import (
    clear_files,
    delete_files,
    load_model,
    get_loaders,
    get_seg_acc,
    # get_seg_acc_pic,
)

# from model.unet3d.model_512 import Generator
from model.encoder_class.model import Generator

# 超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
isLinux = False
MODEL_PATH = "checkpoint/my_checkpoint_epoch_750.pth.tar"

# 测试集路径
TEST_DIR = r"data/test"

print("---------- Load trained model")
# 加载生成器
generator = Generator().to(DEVICE)
load_model(checkpoint=torch.load(MODEL_PATH),
           model=generator,
           model_state_dict_name="state_dict_G")

test_loader = get_loaders(data_dir=TEST_DIR,
                          batch_size=BATCH_SIZE,
                          isLinux=isLinux,
                          isShuffle=False)

# 计算测试集指标
print('Calculate indexes in test dataset...')
test_result_dict = get_seg_acc(test_loader, generator, device=DEVICE)
print('测试集整体指标：', test_result_dict)

# sys.exit(0)

# 可视化测试集结果
print('Visualized in test dataset...')
for i, data in enumerate(test_loader):
    print(i, " / ", len(test_loader))
    # 添加通道维度: [N, 112, 112, 112] -> [N, 1, 112, 112, 112]
    images = data[0].unsqueeze(1).to(DEVICE)  # image: [1, 1, 112, 112, 112]
    gt = data[1].unsqueeze(1).to(DEVICE)  # gt: [1, 1, 112, 112, 112]
    affine = data[2].squeeze(0).numpy()  # 仿射矩阵: numpy.ndarray (4, 4)
    no = data[3][0]  # 检查号: tuple中取出str

    with torch.no_grad():
        pred = torch.sigmoid(generator(images))  # [N, 1, 112, 112, 112]
        pred = (pred > 0.5).float()  # 二值化
        pred = pred.squeeze(dim=0).squeeze(dim=0)  # [112, 112, 112]
        pred = pred.detach().cpu().numpy()  # (112, 112, 112)

        # 计算测试集每条数据的指标
        # test_result_dict = get_seg_acc_pic(gt=gt, pred=pred)
        # dice = test_result_dict["dice"]
        # pre = test_result_dict["pre"]
        # rec = test_result_dict["rec"]
        # fpr = test_result_dict["fpr"]  # 假阳性率
        # fnr = test_result_dict["fnr"]  # 假阴性率
        # print(
        #     f"{no}: --- dice:{dice:.6f}  pre:{pre:.6f}  rec:{rec:.6f}  fpr:{fpr:.6f}  fnr{fnr:.6f}"
        # )

        pred = nib.Nifti1Image(pred, affine)  # 创建一个新的NiBabel图像对象

        # 保存到gt同一路径下
        nib.save(pred, 'data/test/' + no + '/label/' + no + '_label_pred.nii.gz')  # 可视化结果时，只输出预测标签
