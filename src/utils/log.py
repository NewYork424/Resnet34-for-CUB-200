import logging
import os
import sys
from datetime import datetime

def setup_logger(output_root: str, name: str = "ML_Project"):
    """
    配置统一的日志记录器
    1. 创建 logs/name_timestamp/ 文件夹
    2. 在该文件夹下创建 training.log
    3. 返回 logger 和 该次运行的输出目录
    """
    # 生成带时间戳的运行目录名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir_name = f"{name}_{timestamp}"
    run_dir = os.path.join(output_root, run_dir_name)
    
    # 创建目录
    os.makedirs(run_dir, exist_ok=True)
    
    # 日志文件路径
    log_file = os.path.join(run_dir, "training.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = [] # 清除旧handler

    # 格式
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # 文件输出
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    logger.info(f"Run directory: {run_dir}")
    
    return logger, run_dir