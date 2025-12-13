import argparse
import os
import sys
import yaml
import json

# 将 src 目录添加到 python path，确保可以导入子模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.log import setup_logger

def log_detailed_config(logger, cfg):
    """记录详细的配置信息"""
    logger.info("=" * 80)
    logger.info("==== TRAINING CONFIGURATION ====")
    logger.info("=" * 80)
    logger.info("\n[Configuration]")
    # 使用 json dumps 格式化输出字典，确保易读
    logger.info(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
    logger.info("=" * 80)

def load_config(path):
    """加载配置文件，支持相对路径查找"""
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
    parser = argparse.ArgumentParser(description="Unified Bird Classification Entry Point")
    parser.add_argument('--config', type=str, default='config.yml', help='Path to config.yml')
    args = parser.parse_args()

    # 1. 加载配置
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # 2. 初始化日志和运行目录
    log_root = cfg.get('output_dir', './logs')
    if not os.path.isabs(log_root):
        project_root = os.path.dirname(current_dir)
        log_root = os.path.join(project_root, log_root)
        
    model_name = cfg.get('model', 'unknown').lower()
    task_name = cfg.get('task', 'train').lower()
    
    # [修改] 获取 logger 和 run_dir
    logger, run_dir = setup_logger(log_root, name=f"{model_name}_{task_name}")
    
    # [关键] 将生成的运行目录回写到 cfg 中，供后续模块使用
    cfg['output_dir'] = run_dir 

    log_detailed_config(logger, cfg)
    
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Selected Model: {model_name} | Task: {task_name}")

    # 3. 根据配置分发任务
    try:
        if model_name == 'dl':
            logger.info(">>> Initializing Deep Learning (ResNet) module...")
            from deep_learning.run_deeplearn import run_from_config
            run_from_config(cfg, logger)
            
        elif model_name == 'svm':
            logger.info(">>> Initializing SVM module...")
            from svm.run_svm import run_svm_task
            run_svm_task(cfg, logger)
            
        elif model_name == 'decision_tree':
            logger.info(">>> Initializing Decision Tree module...")
            from decision_tree_model.run_tree import run_tree_task
            run_tree_task(cfg, logger)
            
        elif model_name == 'linear':
            logger.info(">>> Initializing Linear Model module...")
            from linear_model.run_linear import run_linear_task
            run_linear_task(cfg, logger)

        elif model_name == 'all':
            logger.info(">>> Running All Models Sequentially...")
            from deep_learning.run_deeplearn import run_from_config
            from svm.run_svm import run_svm_task
            from decision_tree_model.run_tree import run_tree_task
            from linear_model.run_linear import run_linear_task
            
            # SVM
            logger.info(">>> Running SVM module...")
            run_svm_task(cfg, logger)
            
            # Decision Tree
            logger.info(">>> Running Decision Tree module...")
            run_tree_task(cfg, logger)
            
            # Linear Model
            logger.info(">>> Running Linear Model module...")
            run_linear_task(cfg, logger)
            
            # Deep Learning
            logger.info(">>> Running Deep Learning (ResNet) module...")
            run_from_config(cfg, logger)
            
        else:
            valid_models = ['resnet', 'svm', 'decision_tree', 'linear']
            logger.error(f"Unknown model '{model_name}' in config. Valid options: {valid_models}")
            sys.exit(1)
            
    except ImportError as e:
        logger.error(f"Module import failed: {e}")
        logger.error("Please ensure your folder structure matches the project requirements.")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Runtime error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()