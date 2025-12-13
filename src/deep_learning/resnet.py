import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

# 规范化常量
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ===== 网络结构 (保持不变) =====

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, norm_layer=nn.BatchNorm2d, attention: str = "coord"):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        
        if attention == "se":
            self.se = SqueezeExcitation(planes)
        elif attention == "coord":
            self.se = CoordAttention(planes, planes)
        else:
            self.se = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.se is not None:
            out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet34Backbone(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, attention: str = "coord", pool: str = "gem", gem_p: float = 3.0):
        super().__init__()
        block_config = [3, 4, 6, 3]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=block_config[0], stride=1, norm_layer=norm_layer, attention=attention)
        self.layer2 = self._make_layer(128, blocks=block_config[1], stride=2, norm_layer=norm_layer, attention=attention)
        self.layer3 = self._make_layer(256, blocks=block_config[2], stride=2, norm_layer=norm_layer, attention=attention)
        self.layer4 = self._make_layer(512, blocks=block_config[3], stride=2, norm_layer=norm_layer, attention=attention)

        self.avgpool = GeM(p=gem_p) if pool == "gem" else nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int, norm_layer, attention: str):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                norm_layer(planes * BasicBlock.expansion),
            )
        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, norm_layer, attention=attention))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, 1, None, norm_layer, attention=attention))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

class BackboneClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: bool = False, attention: str = "coord", pool: str = "gem", dropout: float = 0.1, gem_p: float = 3.0):
        super().__init__()
        self.backbone = ResNet34Backbone(attention=attention, pool=pool, gem_p=gem_p)
        # [修改] 冻结骨干网络逻辑
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        if return_features:
            x_stem = self.backbone.conv1(x)
            x_stem = self.backbone.bn1(x_stem)
            x_stem = self.backbone.relu(x_stem)
            x_stem = self.backbone.maxpool(x_stem)
            x1 = self.backbone.layer1(x_stem)
            x2 = self.backbone.layer2(x1)
            x3 = self.backbone.layer3(x2)
            x4 = self.backbone.layer4(x3)
            feat = self.backbone.avgpool(x4)
            feat = torch.flatten(feat, 1)
            feat = self.dropout(feat)
            logits = self.classifier(feat)
            return logits, x3
        else:
            feat = self.backbone(x)
            feat = torch.flatten(feat, 1)
            feat = self.dropout(feat)
            logits = self.classifier(feat)
            return logits

# ... (GeM, CoordAttention, SqueezeExcitation, SaliencyGuidedLoss 保持不变) ...
# ===== 辅助模块 =====
class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.nn.functional.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)

class CoordAttention(nn.Module):
    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class SaliencyGuidedLoss(nn.Module):
    def __init__(self, alpha: float = 0.15, kernel_size: int = 5):
        super().__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
    
    def _compute_saliency_map(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weights = torch.tensor([0.299, 0.587, 0.114], device=images.device).view(1, 3, 1, 1)
            img_gray = (images * weights).sum(dim=1, keepdim=True)
            kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=images.device) / (self.kernel_size ** 2)
            local_mean = torch.nn.functional.conv2d(img_gray, kernel, padding=self.padding)
            local_mean_sq = torch.nn.functional.conv2d(img_gray ** 2, kernel, padding=self.padding)
            local_var = (local_mean_sq - local_mean ** 2).clamp(min=0)
            batch_size = local_var.size(0)
            saliency_map = torch.zeros_like(local_var)
            for i in range(batch_size):
                s_min = local_var[i].min()
                s_max = local_var[i].max()
                if s_max - s_min > 1e-6:
                    saliency_map[i] = (local_var[i] - s_min) / (s_max - s_min)
                else:
                    saliency_map[i] = 0.5
        return saliency_map
    
    def _compute_feature_attention(self, features: torch.Tensor) -> torch.Tensor:
        attention = features.pow(2).mean(dim=1, keepdim=True).sqrt()
        batch_size = attention.size(0)
        normalized_attention = torch.zeros_like(attention)
        for i in range(batch_size):
            a_min = attention[i].min().item()
            a_max = attention[i].max().item()
            if a_max - a_min > 1e-6:
                normalized_attention[i] = (attention[i] - a_min) / (a_max - a_min)
            else:
                normalized_attention[i] = 0.5
        return normalized_attention
    
    def forward(self, features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        saliency_map = self._compute_saliency_map(images)
        if saliency_map.shape[-2:] != features.shape[-2:]:
            saliency_map = torch.nn.functional.interpolate(saliency_map, size=features.shape[-2:], mode='bilinear', align_corners=False)
        feat_attention = self._compute_feature_attention(features)
        loss = torch.nn.functional.mse_loss(feat_attention, saliency_map.detach())
        return self.alpha * loss

# ===== 训练器 =====

class DeepLearningClassifier:
    """
    ResNet-34 监督学习训练器
    """
    def __init__(self, cfg: dict, logger, train_loader, val_loader, num_classes: int):
        self.cfg = cfg
        self.logger = logger
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 配置参数
        dl_cfg = cfg['DL']
        self.device = cfg.get("device", "cuda")
        self.epochs = dl_cfg.get("epochs", 300)
        self.save_path = dl_cfg.get("save_path", "best_model.pth")
        self.use_saliency_loss = dl_cfg.get("use_saliency_loss", True)
        
        _set_seed(cfg.get("seed", 42))

        # 模型构建
        self.model = BackboneClassifier(
            num_classes=num_classes,
            freeze_backbone=dl_cfg.get("freeze_backbone", False), 
            attention=dl_cfg.get("attention", "coord"),
            pool=dl_cfg.get("pool", "gem"),
            dropout=dl_cfg.get("dropout", 0.1),
            gem_p=dl_cfg.get("gem_p", 3.0),
        ).to(self.device)
        
        # 加载预训练权重
        pretrained = dl_cfg.get("pretrained_backbone")
        if pretrained and os.path.exists(pretrained):
            self.logger.info(f"Loading pretrained backbone from: {pretrained}")
            try:
                ckpt = torch.load(pretrained, map_location=self.device)
                # 兼容 MoCo 预训练的 key (encoder_q.xxx)
                state_dict = ckpt['encoder_q'] if 'encoder_q' in ckpt else ckpt
                # 移除 encoder_q. 前缀如果存在
                state_dict = {k.replace("encoder_q.", ""): v for k, v in state_dict.items()}
                
                missing, unexpected = self.model.backbone.load_state_dict(state_dict, strict=False)
                self.logger.info(f"✓ Pretrained backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            except Exception as e:
                self.logger.error(f"Failed to load pretrained backbone: {e}")

        # 优化器与损失
        self.criterion = nn.CrossEntropyLoss(label_smoothing=dl_cfg.get("label_smoothing", 0.1))
        
        # [修改] 显著性损失参数化
        if self.use_saliency_loss:
            alpha = dl_cfg.get("saliency_alpha", 0.15)
            k_size = dl_cfg.get("saliency_kernel_size", 5)
            self.saliency_loss_fn = SaliencyGuidedLoss(alpha=alpha, kernel_size=k_size)
        else:
            self.saliency_loss_fn = None
        
        # [修改] 优化器参数分组 (区分 weight_decay)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay.append(p)
            else:
                decay.append(p)
        
        self.optimizer = optim.AdamW(
            [{"params": decay, "weight_decay": dl_cfg.get("weight_decay", 1e-4)},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=dl_cfg.get("lr", 0.0003)
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        self.scaler = GradScaler(enabled=(self.device == "cuda"))
        self.best_val_acc = 0.0

    def fit_one_epoch(self):
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for imgs, labels in self.train_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            imgs_orig = imgs.clone() if self.saliency_loss_fn else None

            with autocast(device_type=self.device, enabled=(self.device=="cuda")):
                if self.saliency_loss_fn:
                    logits, layer3_feat = self.model(imgs, return_features=True)
                    sal_loss = self.saliency_loss_fn(layer3_feat, imgs_orig)
                else:
                    logits = self.model(imgs)
                    sal_loss = 0.0
                
                cls_loss = self.criterion(logits, labels)
                loss = cls_loss + sal_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += cls_loss.item() * labels.size(0)

        return total_loss / total, total_correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        for imgs, labels in self.val_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)
        return total_loss / total, total_correct / total

    def train(self):
        self.logger.info(f"Starting Supervised Training for {self.epochs} epochs...")
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
                self.logger.info(f"Best checkpoint updated: {save_full_path}")
            
            # 格式化日志输出
            self.logger.info(f"Epoch {ep}/{self.epochs} | T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}{improved}")

