"""
MoCo v2 对比学习预训练
用于让模型关注鸟类主体特征而非背景
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from .resnet import ResNet34Backbone, _set_seed


class ContrastiveDatasetWrapper(Dataset):
    """
    包装 UnifiedBirdDataset，忽略标签，返回两个增强视图
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 修改：从 samples 中提取图像路径（四元组的第一个元素）
        img_path, _, _, _ = self.dataset.samples[idx]
        
        # 重新加载图像以确保是 PIL Image
        from PIL import Image
        img = Image.open(img_path).convert("RGB")
        
        # 生成两个不同的增强视图
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2


class MoCo(nn.Module):
    """MoCo v2 简化版"""
    def __init__(self, backbone, dim=128, K=2048, m=0.999, T=0.1):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)
        
        self.projector_q = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, dim))
        self.projector_k = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, dim))
        
        for param in self.encoder_k.parameters(): param.requires_grad = False
        for param in self.projector_k.parameters(): param.requires_grad = False
        
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _momentum_update(self):
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1. - self.m)
        for pq, pk in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            pk.data = pk.data * self.m + pq.data * (1. - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            rem = self.K - ptr
            self.queue[:, ptr:] = keys[:rem].T
            self.queue[:, :batch_size - rem] = keys[rem:].T
        self.queue_ptr[0] = (ptr + batch_size) % self.K
    
    def forward(self, im_q, im_k):
        q = F.normalize(self.projector_q(torch.flatten(self.encoder_q(im_q), 1)), dim=1)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.projector_k(torch.flatten(self.encoder_k(im_k), 1)), dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        self._dequeue_and_enqueue(k)
        return logits, labels


class ContrastiveTrainer:
    def __init__(self, cfg, logger, train_loader):
        self.cfg = cfg
        self.logger = logger
        self.loader = train_loader
        
        dl_cfg = cfg['DL']
        moco_cfg = dl_cfg['moco']
        self.device = cfg.get("device", "cuda")
        self.epochs = moco_cfg.get("epochs", 200)
        
        _set_seed(cfg.get("seed", 42))
        
        backbone = ResNet34Backbone(
            attention=dl_cfg.get("attention", "coord"),
            pool=dl_cfg.get("pool", "gem"),
            gem_p=dl_cfg.get("gem_p", 3.0)
        )
        
        self.model = MoCo(
            backbone=backbone,
            dim=moco_cfg.get("dim", 128),
            K=moco_cfg.get("k", 2048),
            m=moco_cfg.get("m", 0.999),
            T=moco_cfg.get("t", 0.1)
        ).to(self.device)
        
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=moco_cfg.get("lr", 0.03), 
            momentum=moco_cfg.get("momentum", 0.9), 
            weight_decay=dl_cfg.get("weight_decay", 1e-4)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.logger.info(f"Starting Contrastive Pretraining for {self.epochs} epochs...")
        save_dir = self.cfg['output_dir']

        best_loss = float('inf')
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, total_acc = 0.0, 0.0
            
            for im_q, im_k in self.loader:
                im_q, im_k = im_q.to(self.device), im_k.to(self.device)
                logits, labels = self.model(im_q, im_k)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                acc = (torch.argmax(logits, dim=1) == labels).float().mean().item()
                total_acc += acc
            
            self.scheduler.step()
            avg_loss = total_loss / len(self.loader)
            avg_acc = total_acc / len(self.loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
            
            self.logger.info(f"Epoch {epoch}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
            
            # 保存准确率最佳的权重，确保可二次加载
            if avg_loss == best_loss:
                path = os.path.join(save_dir, f"best_pretrain_weight.pth")
                torch.save(
                    {'epoch': epoch,
                    'encoder_q': self.model.encoder_q.state_dict(),
                    'projector_q': self.model.projector_q.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'loss': avg_loss,
                    'acc': avg_acc,
                    'config': self.cfg,
                }, path)
                self.logger.info(f"Saved checkpoint to {path}")