import argparse
import os
import sys
import yaml
import json

# 添加 src 目录到 python path，方便导入
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.log import setup_logger

def log_detailed_config(logger, cfg):
    logger.info("=" * 80)
    logger.info("==== TRAINING CONFIGURATION ====")
    logger.info("=" * 80)
    logger.info("\n[Configuration]")
    logger.info(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    logger.info("=" * 80)

def load_config(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    project_root = os.path.dirname(current_dir)
    root_path = os.path.join(project_root, path)
    if os.path.exists(root_path):
        with open(root_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found at: {path} or {root_path}")

def main():
    parser = argparse.ArgumentParser(description="Bird Classification Entry")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config.yml')
    args = parser.parse_args()

    # 加载配置
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 日志和运行目录
    log_root = cfg.get('output_dir', './logs')
    if not os.path.isabs(log_root):
        project_root = os.path.dirname(current_dir)
        log_root = os.path.join(project_root, log_root)
    model_name = cfg.get('model', 'unknown').lower()
    task_name = cfg.get('task', 'train').lower()
    logger, run_dir = setup_logger(log_root, name=f"{model_name}_{task_name}")
    cfg['output_dir'] = run_dir

    log_detailed_config(logger, cfg)
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Selected Model: {model_name} | Task: {task_name}")

    # 分发任务
    try:
        if model_name == 'dl':
            logger.info(">>> Deep Learning (ResNet) ...")
            from deep_learning.run_deeplearn import run_from_config
            run_from_config(cfg, logger)
        elif model_name == 'svm':
            logger.info(">>> SVM ...")
            from svm.run_svm import run_svm_task
            run_svm_task(cfg, logger)
        elif model_name == 'decision_tree':
            logger.info(">>> Decision Tree ...")
            from decision_tree_model.run_tree import run_tree_task
            run_tree_task(cfg, logger)
        elif model_name == 'linear':
            logger.info(">>> Linear Model ...")
            from linear_model.run_linear import run_linear_task
            run_linear_task(cfg, logger)
        elif model_name == 'all':
            logger.info(">>> Running All Models ...")
            from deep_learning.run_deeplearn import run_from_config
            from svm.run_svm import run_svm_task
            from decision_tree_model.run_tree import run_tree_task
            from linear_model.run_linear import run_linear_task
            logger.info(">>> SVM ...")
            run_svm_task(cfg, logger)
            logger.info(">>> Decision Tree ...")
            run_tree_task(cfg, logger)
            logger.info(">>> Linear Model ...")
            run_linear_task(cfg, logger)
            logger.info(">>> Deep Learning (ResNet) ...")
            run_from_config(cfg, logger)
        else:
            valid_models = ['resnet', 'svm', 'decision_tree', 'linear']
            logger.error(f"Unknown model '{model_name}' in config. Valid options: {valid_models}")
            sys.exit(1)
    except ImportError as e:
        logger.error(f"Module import failed: {e}")
        logger.error("Check your folder structure.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()