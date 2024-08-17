import numpy as np
import os
import sys
import nibabel as nib
from torch.utils.data import Dataset


class SegDataset(Dataset):
    def __init__(self,
                 data_dir=r'D:\Me\XBMU_Lab\07 Dataset\02_PET\PET\04_all_dataset_suv\train'):
        """
        :param data_dir: 数据路径
        """
        self.dataDir = data_dir  # 训练集、测试集路径
        self.datas = os.listdir(data_dir)  # 全部子文件夹名

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        """
        :param index: 数据下标
        :return: 图片，分割标签，仿射变换矩阵，检查号
        """
        no = self.datas[index]  # str
        img_path = self.dataDir + '/' + no + '/image/' + no + '_SUV.nii.gz'
        label_path = self.dataDir + '/' + no + '/label/' + no + '_label.nii.gz'

        # 图片（单个）
        nifti_img = nib.load(img_path)
        img = nifti_img.get_fdata().astype(np.float32)  # numpy.ndarray

        # 分割标签（单个）
        nifti_label = nib.load(label_path)
        label = nifti_label.get_fdata().astype(np.float32)  # numpy.ndarray

        return img, label, nifti_img.affine, no


if __name__ == '__main__':
    # seg_data = SegDataset(data_dir=r"./data/train")
    seg_data = SegDataset(
        data_dir=r"D:\Me\XBMU_Lab\07 Dataset\02_PET\PET\04_all_dataset_suv\train")

    print(type(seg_data[0][0]))  # img (numpy.ndarray)
    print(type(seg_data[0][1]))  # label (numpy.ndarray)

    print(seg_data[0][0].shape)  # (336, 432, 400)
    print(seg_data[0][1].shape)  # (336, 432, 400)

    affine = seg_data[0][2]  # nifti_img.affine
    no = seg_data[0][3]  # tuple 中取出 str

    print('no: ', no)  # 000007210
    print('type(no): ', type(no))  # str

    # print('affine: ', affine)
    # print('type(affine): ', type(affine))  # numpy.ndarray
    # print('affine.shape: ', affine.shape)  # (4, 4)

    print(seg_data.__len__())  # 21
