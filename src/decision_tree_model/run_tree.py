import os
import json
import pickle 
from .decision_tree import CARTDecisionTreeClassifier, C45DecisionTreeClassifier
from utils.dataset import UnifiedBirdDataset

def run_tree_task(cfg, logger):
    tree_cfg = cfg['decision_tree']
    data_cfg = cfg['data']
    output_dir = cfg['output_dir'] 
    
    feature_type = tree_cfg.get('feature_type', 'attr')
    logger.info(f"Running Decision Tree with feature_type={feature_type}")

    # 加载数据
    train_ds = UnifiedBirdDataset(data_cfg['root'], 'train', img_size=data_cfg['img_size'], class_num=tree_cfg.get('class_num', 10))
    val_ds = UnifiedBirdDataset(data_cfg['root'], 'val', img_size=data_cfg['img_size'], class_num=tree_cfg.get('class_num', 10))
    
    X_train, y_train = train_ds.get_sklearn_data(feature_type)
    X_test, y_test = val_ds.get_sklearn_data(feature_type)

    # 训练
    logger.info("Training Decision Tree...")
    
    # [新增] 读取剪枝配置
    common_params = {
        "max_depth": tree_cfg.get('max_depth', None),
        "min_samples_split": tree_cfg.get('min_samples_split', 2),
        "min_samples_leaf": tree_cfg.get('min_samples_leaf', 1),
        "min_impurity_decrease": tree_cfg.get('min_impurity_decrease', 0.0),
        "random_state": tree_cfg.get('seed', 42),
        "pre_pruning": tree_cfg.get('pre_pruning', True),   # 默认开启预剪枝
        "post_pruning": tree_cfg.get('post_pruning', False), # 默认关闭后剪枝
        "val_ratio": tree_cfg.get('val_ratio', 0.2)
    }

    if tree_cfg['criterion'] == 'c45':
        clf = C45DecisionTreeClassifier(**common_params)
    else:
        clf = CARTDecisionTreeClassifier(**common_params)
        
    clf.fit(X_train, y_train)
    
    # 评估
    acc = clf.score(X_test, y_test)
    logger.info(f"Decision Tree Accuracy: {acc:.4f}")

    # 保存结果
    model_path = os.path.join(output_dir, "tree_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info(f"Model saved to {model_path}")

    try:
        tree_dict = clf.export_dict()
        json_path = os.path.join(output_dir, "tree_structure.json")
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Tree structure saved to {json_path}")
    except Exception as e:
        logger.warning(f"Failed to export tree structure: {e}")

    with open(os.path.join(output_dir, "results.txt"), "w") as f:
        f.write(f"Model: Decision Tree ({tree_cfg['criterion']})\n")
        f.write(f"Pre-pruning: {common_params['pre_pruning']}\n")
        f.write(f"Post-pruning: {common_params['post_pruning']}\n")
        f.write(f"Accuracy: {acc:.4f}\n")