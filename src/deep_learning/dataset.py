# 重新实现为 torch.utils.data.Dataset，支持 transforms 和可选属性返回
import os
from typing import List, Tuple, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset

class DatasetDeepLearning(Dataset):
    def __init__(self, data_path: str, transform=None, return_attrs: bool = False, img_ext: str = ".jpg", attr_ext: str = ".pt"):
        self.data_path = data_path
        self.transform = transform
        self.return_attrs = return_attrs
        self.img_ext = img_ext
        self.attr_ext = attr_ext

        self.samples: List[Tuple[str, int, Optional[str]]] = []
        self.class_names: List[str] = []

        self._scan()

    def _scan(self):
        # 扫描类别目录
        class_dirs = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        class_dirs.sort()

        for class_dir in class_dirs:
            try:
                idx_str, name = class_dir.split(".", 1)
                class_index = int(idx_str) - 1
            except Exception:
                continue

            while len(self.class_names) <= class_index:
                self.class_names.append("")
            self.class_names[class_index] = name

            abs_class_dir = os.path.join(self.data_path, class_dir)
            for fname in os.listdir(abs_class_dir):
                if not fname.lower().endswith(self.img_ext):
                    continue
                img_path = os.path.join(abs_class_dir, fname)
                stem = os.path.splitext(fname)[0]
                attr_path = os.path.join(abs_class_dir, stem + self.attr_ext)
                if not os.path.exists(attr_path):
                    if self.return_attrs:
                        raise FileNotFoundError(f"Attr file not found for image: {img_path}")
                    attr_path = None
                self.samples.append((img_path, class_index, attr_path))

        if not self.samples:
            raise RuntimeError(f"No samples found under: {self.data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, attr_path = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_attrs and attr_path is not None:
            attr = torch.load(attr_path)
            return img, label, attr
        return img, label

    def get_imgs(self):
        # 不再一次性加载到内存，建议用 DataLoader
        return None

    def get_attrs(self):
        return None

    def get_labels(self):
        # 返回类别名列表
        return self.class_names

    def get_data(self):
        # 返回文件路径和标签，便于兼容
        img_paths = [s[0] for s in self.samples]
        labels = torch.tensor([s[1] for s in self.samples], dtype=torch.long)
        attr_paths = [s[2] for s in self.samples]
        return img_paths, attr_paths, labels
