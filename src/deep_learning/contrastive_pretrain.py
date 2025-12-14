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
from PIL import Image
from .resnet import ResNet34Backbone, _set_seed


class ContrastiveDatasetWrapper(Dataset):
    """包装数据集，生成两个增强视图用于对比学习"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path = self.dataset.samples[idx][0]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), self.transform(img)


class MoCo(nn.Module):
    """MoCo v2 简化实现"""
    def __init__(self, backbone, dim=128, K=2048, m=0.999, T=0.1):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        
        # query encoder和key encoder
        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)
        
        # 投影头，将512维特征映射到128维
        self.projector_q = nn.Sequential(
            nn.Linear(512, 512), 
            nn.ReLU(), 
            nn.Linear(512, dim)
        )
        self.projector_k = copy.deepcopy(self.projector_q)
        
        # key encoder不需要梯度
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.projector_k.parameters():
            p.requires_grad = False
        
        # 负样本队列
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def _update_key_encoder(self):
        """动量更新key encoder"""
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)
        for p_q, p_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            p_k.data = p_k.data * self.m + p_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _update_queue(self, keys):
        """更新负样本队列"""
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + bs <= self.K:
            self.queue[:, ptr:ptr + bs] = keys.T
        else:
            # 队列满了就从头开始覆盖
            rem = self.K - ptr
            self.queue[:, ptr:] = keys[:rem].T
            self.queue[:, :bs - rem] = keys[rem:].T
        
        self.queue_ptr[0] = (ptr + bs) % self.K
    
    def forward(self, im_q, im_k):
        # query编码
        q = self.encoder_q(im_q)
        q = torch.flatten(q, 1)
        q = F.normalize(self.projector_q(q), dim=1)
        
        # key编码，不计算梯度
        with torch.no_grad():
            self._update_key_encoder()
            k = self.encoder_k(im_k)
            k = torch.flatten(k, 1)
            k = F.normalize(self.projector_k(k), dim=1)
        
        # 计算正样本和负样本的相似度
        pos_sim = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        neg_sim = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        self._update_queue(k)
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
        
        # 构建backbone
        backbone = ResNet34Backbone(
            attention=dl_cfg.get("attention", "coord"),
            pool=dl_cfg.get("pool", "gem"),
            gem_p=dl_cfg.get("gem_p", 3.0)
        )
        
        # 构建MoCo模型
        self.model = MoCo(
            backbone=backbone,
            dim=moco_cfg.get("dim", 128),
            K=moco_cfg.get("k", 2048),
            m=moco_cfg.get("m", 0.999),
            T=moco_cfg.get("t", 0.1)
        ).to(self.device)
        
        # 优化器，MoCo原文用的SGD
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=moco_cfg.get("lr", 0.03), 
            momentum=0.9, 
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs
        )
        self.criterion = nn.CrossEntropyLoss()
        self.best_loss = float('inf')

    def train(self):
        self.logger.info(f"Start contrastive pretraining, total {self.epochs} epochs")
        save_dir = self.cfg['output_dir']
        
        for ep in range(1, self.epochs + 1):
            self.model.train()
            running_loss, running_acc = 0.0, 0.0
            
            for im_q, im_k in self.loader:
                im_q = im_q.to(self.device)
                im_k = im_k.to(self.device)
                
                logits, labels = self.model(im_q, im_k)
                loss = self.criterion(logits, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                acc = (logits.argmax(1) == labels).float().mean().item()
                running_acc += acc
            
            self.scheduler.step()
            
            avg_loss = running_loss / len(self.loader)
            avg_acc = running_acc / len(self.loader)
            
            msg = f"Epoch {ep}/{self.epochs} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}"
            
            # 保存最优模型
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                ckpt_path = os.path.join(save_dir, "best_pretrain_weight.pth")
                torch.save({
                    'epoch': ep,
                    'encoder_q': self.model.encoder_q.state_dict(),
                    'projector_q': self.model.projector_q.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'loss': avg_loss
                }, ckpt_path)
                msg += " [saved]"
            
            self.logger.info(msg)