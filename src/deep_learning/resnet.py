import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# 规范化常量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)


class GeM(nn.Module):
    """广义平均池化，p>1时更关注激活值较大的区域"""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class SqueezeExcitation(nn.Module):
    """SE注意力：通道加权，参数量约为原网络1%"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.fc(self.pool(x))


class CoordAttention(nn.Module):
    """坐标注意力：保留空间位置信息，适合细粒度任务"""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)
    
    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_w * a_h


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 norm_layer=nn.BatchNorm2d, attention="coord"):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        
        # 根据配置选择注意力模块
        if attention == "se":
            self.se = SqueezeExcitation(planes)
        elif attention == "coord":
            self.se = CoordAttention(planes, planes)
        else:
            self.se = None
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet34Backbone(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, attention="coord", 
                 pool="gem", gem_p=3.0):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet34配置：[3,4,6,3]
        self.layer1 = self._make_layer(64, 3, 1, norm_layer, attention)
        self.layer2 = self._make_layer(128, 4, 2, norm_layer, attention)
        self.layer3 = self._make_layer(256, 6, 2, norm_layer, attention)
        self.layer4 = self._make_layer(512, 3, 2, norm_layer, attention)
        
        self.avgpool = GeM(p=gem_p) if pool == "gem" else nn.AdaptiveAvgPool2d((1, 1))
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _make_layer(self, planes, blocks, stride, norm_layer, attention):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes)
            )
        layers = [BasicBlock(self.inplanes, planes, stride, downsample, norm_layer, attention)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None, norm_layer, attention))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x)


class BackboneClassifier(nn.Module):
    def __init__(self, num_classes, freeze_backbone=False, attention="coord", 
                 pool="gem", dropout=0.1, gem_p=3.0):
        super().__init__()
        self.backbone = ResNet34Backbone(attention=attention, pool=pool, gem_p=gem_p)
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(512, num_classes)
    
    def forward(self, x, return_features=False):
        if return_features:
            x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x3 = self.backbone.layer3(x)
            x = self.backbone.layer4(x3)
            feat = torch.flatten(self.backbone.avgpool(x), 1)
            logits = self.classifier(self.dropout(feat))
            return logits, x3
        else:
            feat = torch.flatten(self.backbone(x), 1)
            return self.classifier(self.dropout(feat))


class SaliencyGuidedLoss(nn.Module):
    """无标注显著性损失：基于局部方差生成伪标签，引导网络关注高频纹理区域"""
    def __init__(self, alpha=0.15, kernel_size=5):
        super().__init__()
        self.alpha = alpha
        self.padding = kernel_size // 2
        self.register_buffer('kernel', torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2))
        self.register_buffer('rgb_weights', torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
    
    def _get_saliency_map(self, images):
        """局部方差法计算显著性"""
        with torch.no_grad():
            gray = (images * self.rgb_weights).sum(1, keepdim=True)
            mean = torch.nn.functional.conv2d(gray, self.kernel, padding=self.padding)
            mean_sq = torch.nn.functional.conv2d(gray ** 2, self.kernel, padding=self.padding)
            var = (mean_sq - mean ** 2).clamp(min=0)
            
            # 按batch归一化，避免全局统计偏差
            saliency = torch.zeros_like(var)
            for i in range(var.size(0)):
                v_min, v_max = var[i].min(), var[i].max()
                if v_max - v_min > 1e-6:
                    saliency[i] = (var[i] - v_min) / (v_max - v_min)
                else:
                    saliency[i] = 0.5
        return saliency
    
    def _get_feature_attention(self, features):
        """特征图能量归一化"""
        attn = features.pow(2).mean(1, keepdim=True).sqrt()
        normed = torch.zeros_like(attn)
        for i in range(attn.size(0)):
            a_min, a_max = attn[i].min(), attn[i].max()
            if a_max - a_min > 1e-6:
                normed[i] = (attn[i] - a_min) / (a_max - a_min)
            else:
                normed[i] = 0.5
        return normed
    
    def forward(self, features, images):
        sal = self._get_saliency_map(images)
        if sal.shape[-2:] != features.shape[-2:]:
            sal = torch.nn.functional.interpolate(
                sal, size=features.shape[-2:], mode='bilinear', align_corners=False
            )
        feat_attn = self._get_feature_attention(features)
        return self.alpha * torch.nn.functional.mse_loss(feat_attn, sal.detach())


class DeepLearningClassifier:
    """监督学习训练流程封装"""
    def __init__(self, cfg, logger, train_loader, val_loader, num_classes):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        dl_cfg = cfg['DL']
        self.device = cfg.get("device", "cuda")
        self.epochs = dl_cfg.get("epochs", 300)
        self.save_path = dl_cfg.get("save_path", "best_model.pth")
        self.use_sal_loss = dl_cfg.get("use_saliency_loss", True)
        
        _set_seed(cfg.get("seed", 42))
        
        # 构建模型
        self.model = BackboneClassifier(
            num_classes=num_classes,
            freeze_backbone=dl_cfg.get("freeze_backbone", False),
            attention=dl_cfg.get("attention", "coord"),
            pool=dl_cfg.get("pool", "gem"),
            dropout=dl_cfg.get("dropout", 0.1),
            gem_p=dl_cfg.get("gem_p", 3.0)
        ).to(self.device)
        
        # 加载预训练权重，兼容MoCo格式
        pretrained_path = dl_cfg.get("pretrained_backbone")
        if pretrained_path and os.path.exists(pretrained_path):
            self.logger.info(f"Loading pretrained backbone: {pretrained_path}")
            try:
                ckpt = torch.load(pretrained_path, map_location=self.device)
                state_dict = ckpt.get('encoder_q', ckpt)
                state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
                missing, unexpected = self.model.backbone.load_state_dict(state_dict, strict=False)
                self.logger.info(f"Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
            except Exception as e:
                self.logger.error(f"Failed to load pretrained weights: {e}")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss(label_smoothing=dl_cfg.get("label_smoothing", 0.1))
        if self.use_sal_loss:
            self.sal_loss_fn = SaliencyGuidedLoss(
                alpha=dl_cfg.get("saliency_alpha", 0.15),
                kernel_size=dl_cfg.get("saliency_kernel_size", 5)
            )
        else:
            self.sal_loss_fn = None
        
        # 优化器参数分组，避免bias和LayerNorm的weight_decay
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        
        self.optimizer = optim.AdamW([
            {"params": decay, "weight_decay": dl_cfg.get("weight_decay", 1e-4)},
            {"params": no_decay, "weight_decay": 0.0}
        ], lr=dl_cfg.get("lr", 3e-4))
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler(enabled=(self.device == "cuda"))
        self.best_val_acc = 0.0
    
    def fit_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            with autocast(device_type=self.device, enabled=(self.device == "cuda")):
                if self.sal_loss_fn:
                    logits, feat = self.model(imgs, return_features=True)
                    sal_loss = self.sal_loss_fn(feat, imgs)
                else:
                    logits = self.model(imgs)
                    sal_loss = 0.0
                
                cls_loss = self.criterion(logits, labels)
                loss = cls_loss + sal_loss
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += cls_loss.item() * labels.size(0)
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        return total_loss / total, correct / total
    
    def train(self):
        self.logger.info(f"Starting training for {self.epochs} epochs")
        save_full_path = os.path.join(self.cfg['output_dir'], self.save_path)
        
        for ep in range(1, self.epochs + 1):
            train_loss, train_acc = self.fit_one_epoch()
            val_loss, val_acc = self.evaluate()
            self.scheduler.step()
            
            improved = ""
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_full_path)
                improved = " [best]"
                self.logger.info(f"Checkpoint saved: {save_full_path}")
            
            self.logger.info(
                f"Epoch {ep}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{improved}"
            )

