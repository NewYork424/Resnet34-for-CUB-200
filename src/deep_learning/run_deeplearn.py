import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from .resnet import DeepLearningClassifier, IMAGENET_MEAN, IMAGENET_STD
from .contrastive_pretrain import ContrastiveTrainer, ContrastiveDatasetWrapper
from utils.dataset import UnifiedBirdDataset

def log_model_structure(logger, model):
    logger.info("==== MODEL ARCHITECTURE ====")
    logger.info("=" * 80)
    logger.info(f"\n{model}")
    logger.info("=" * 80)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")
    logger.info("=" * 80)

def run_from_config(global_cfg, logger):
    """
    深度学习任务入口
    """
    dl_cfg = global_cfg['DL']
    data_cfg = global_cfg['data']
    moco_cfg = dl_cfg.get('moco', {})
    mode = dl_cfg.get('mode', 'supervised')
    logger.info(f"Deep Learning Mode: {mode.upper()}")

    if mode == 'pretrain':
        img_size = moco_cfg.get('image_size', 224)
        moco_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        base_ds = UnifiedBirdDataset(data_cfg['root'], 'train', transform=None, class_num=dl_cfg.get('class_num', None))
        train_ds = ContrastiveDatasetWrapper(base_ds, moco_transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=moco_cfg['batch_size'],
            shuffle=True,
            num_workers=data_cfg['num_workers'],
            drop_last=True
        )
        logger.info(f"Pretraining samples: {len(train_ds)}")
        trainer = ContrastiveTrainer(global_cfg, logger, train_loader)
        log_model_structure(logger, trainer.model)
        trainer.train()
    else:
        img_size = dl_cfg.get('img_size', 448)
        use_randaugment = dl_cfg.get("use_randaugment", True)
        ra_num_ops = dl_cfg.get("ra_num_ops", 3)
        ra_magnitude = dl_cfg.get("ra_magnitude", 12)
        train_scale_min = dl_cfg.get("train_scale_min", 0.08)
        random_erasing_p = dl_cfg.get("random_erasing_p", 0.0)

        train_tf_list = [
            transforms.RandomResizedCrop(img_size, scale=(train_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            try:
                ra = transforms.RandAugment(num_ops=ra_num_ops, magnitude=ra_magnitude)
                train_tf_list.append(ra)
            except AttributeError:
                logger.warning("torchvision version too old for RandAugment, using ColorJitter instead.")
                train_tf_list.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8))
        train_tf_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if random_erasing_p > 0:
            re_scale = tuple(dl_cfg.get("random_erasing_scale", [0.02, 0.33]))
            re_ratio = tuple(dl_cfg.get("random_erasing_ratio", [0.3, 3.3]))
            train_tf_list.append(
                transforms.RandomErasing(p=random_erasing_p, scale=re_scale, ratio=re_ratio, value="random")
            )
        train_tf = transforms.Compose(train_tf_list)
        val_tf = transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        train_ds = UnifiedBirdDataset(data_cfg['root'], 'train', transform=train_tf, class_num=dl_cfg.get('class_num', None))
        val_ds = UnifiedBirdDataset(data_cfg['root'], 'val', transform=val_tf, class_num=dl_cfg.get('class_num', None))
        train_loader = DataLoader(train_ds, batch_size=dl_cfg['batch_size'], shuffle=True, num_workers=data_cfg['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=dl_cfg['batch_size'], shuffle=False, num_workers=data_cfg['num_workers'])
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        num_classes = len(train_ds.classes)
        trainer = DeepLearningClassifier(global_cfg, logger, train_loader, val_loader, num_classes)
        log_model_structure(logger, trainer.model)
        trainer.train()