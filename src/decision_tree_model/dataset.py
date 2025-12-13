import os
from sympy import im
import torch
import numpy as np
import pickle as pkl
import PIL.Image as Image

class DatasetTree():
    def __init__(self, data_path, transform=None, img_only=False, attr_only=False):
        self.data_path = data_path
        self.transform = transform
        self.img_only = img_only
        self.attr_only = attr_only
        self.imgs, self.attrs, self.labels, self.labels_map = self.load_data(data_path)
    
    def load_data(self, path):
        # 加载目标地址下的所有文件夹地址
        paths = [os.path.join(path, folder) for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
        imgs = []
        labels = []
        attrs = []
        labels_map = []
        for category_path in paths[:10]:
            img_paths = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith('.jpg')]
            attr_paths = [os.path.join(category_path, attr) for attr in os.listdir(category_path) if attr.endswith('.pt')]
            if len(img_paths) != len(attr_paths):
                raise ValueError(f"Number of images and attributes do not match in {category_path}")

            label = os.path.basename(category_path)
            label_num = int(label.split('.')[0]) - 1  # 从0开始编号
            label_name = label.split('.')[1]
            labels_map.append(label_name)
            labels.extend([label_num] * len(img_paths))

            for img_path, attr_path in zip(img_paths, attr_paths):
                if self.img_only:
                    attr = torch.zeros(1)  # 占位符
                else:
                    attr = torch.load(attr_path)
                    attr = attr.numpy().astype(np.float32)
                
                if self.attr_only:
                    img = torch.zeros(1, 1, 1)  # 占位符
                else:
                    img = Image.open(img_path)
                    # img = torch.tensor(np.asarray(img))
                
                imgs.append(img)
                attrs.append(attr)

        return imgs, attrs, labels, labels_map
    def get_imgs(self):
        return self.imgs
    
    def get_attrs(self):
        return self.attrs

    def get_labels(self):
        return self.labels_map

    def get_data(self):
        """
        获取数据集的图像、标签和属性。
        """
        return self.imgs, self.attrs, self.labels
