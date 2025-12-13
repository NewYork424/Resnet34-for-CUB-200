import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Tuple, Literal

class UnifiedBirdDataset(Dataset):
    """
    统一数据集接口，支持深度学习(Tensor)和传统机器学习(Numpy)
    """
    def __init__(self, root_dir: str, split: str = "train", transform=None, img_size: int = 224, class_num: int = 10):
        self.root = os.path.join(root_dir, split)
        self.transform = transform
        self.img_size = img_size
        self.samples = [] # (img_path, attr_path, label_idx, label_name)
        self.classes = []
        self.class_num = class_num
        
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Data directory not found: {self.root}")
            
        self._scan_directory()

    def _scan_directory(self):
        """扫描目录结构: root/1.cat/xxx.jpg"""
        class_dirs = [d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))]
        # 按数字索引排序
        class_dirs.sort(key=lambda x: int(x.split('.')[0]) if '.' in x else x)

        if self.class_num > len(class_dirs):
            self.class_num = len(class_dirs)

        for d in class_dirs[:self.class_num]:
            try:
                idx_str, name = d.split('.', 1)
                label_idx = int(idx_str) - 1
            except ValueError:
                continue # 跳过不符合格式的文件夹

            # 记录类别名
            while len(self.classes) <= label_idx:
                self.classes.append("")
            self.classes[label_idx] = name

            dir_path = os.path.join(self.root, d)
            for fname in os.listdir(dir_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(dir_path, fname)
                    attr_path = os.path.join(dir_path, os.path.splitext(fname)[0] + '.pt')
                    
                    if not os.path.exists(attr_path):
                        attr_path = None
                    
                    self.samples.append((img_path, attr_path, label_idx, name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """PyTorch 接口: 返回 (img_tensor, label)"""
        img_path, _, label, _ = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        else:
            # 默认简单的转换
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img)
            
        return img, label

    def get_sklearn_data(self, feature_type: Literal['attr', 'image'] = 'attr'):
        """
        Sklearn 接口: 一次性加载所有数据到内存
        Returns: X (numpy array), y (numpy array)
        """
        X_list = []
        y_list = []
        
        print(f"Loading {len(self.samples)} samples for {feature_type}...")
        
        for img_path, attr_path, label, _ in self.samples:
            if feature_type == 'attr':
                if attr_path is None:
                    # 如果缺失属性，用全0填充 (或者抛出异常)
                    feat = np.zeros(2048, dtype=np.float32) # 假设维度
                else:
                    feat = torch.load(attr_path).numpy().astype(np.float32)
                X_list.append(feat)
            else:
                # 加载图片并展平
                img = Image.open(img_path).convert('RGB').resize((self.img_size, self.img_size))
                feat = np.array(img).flatten().astype(np.float32)
                X_list.append(feat)
            
            y_list.append(label)
            
        return np.array(X_list), np.array(y_list)