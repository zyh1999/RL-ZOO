import optuna
from typing import Any
import torch.nn as nn
import glob
import os
import pickle
import sys

# 导入 RL-Zoo 的关键组件
from rl_zoo3 import hyperparams_opt
# 我们需要使用原有的 sample_ppo_params 作为基础
from rl_zoo3.hyperparams_opt import sample_ppo_params, convert_onpolicy_params
from rl_zoo3.train import train

# ==============================================================================
# 自定义 PPO 搜索空间
# ==============================================================================
def sample_ppo_params_custom(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> dict[str, Any]:
    """
    自定义的 PPO 参数采样函数
    - 复用 RL-Zoo 默认搜索逻辑
    - 覆盖特定的几个参数：learning_rate, batch_size, n_steps, normalize_advantage_mean
    - 强制设定 ent_coef = 0 和 separate_optimizers = True
    """
    # --- 复用 hyperparams_opt.py 中的默认范围 (除了我们要覆盖的) ---
    one_minus_gamma = trial.suggest_float("one_minus_gamma", 0.0001, 0.03, log=True)
    one_minus_gae_lambda = trial.suggest_float("one_minus_gae_lambda", 0.0001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    # n_epochs: 改为 1-10 的整数搜索
    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5)
    
    # 网络架构默认逻辑
    # net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # net_arch = {
    #     "tiny": dict(pi=[64], vf=[64]),
    #     "small": dict(pi=[64, 64], vf=[64, 64]),
    #     "medium": dict(pi=[256, 256], vf=[256, 256]),
    # }[net_arch_type]
    
    # 激活函数默认逻辑
    # activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

    # ==========================================================================
    # 下面是你要求的自定义部分
    # ==========================================================================
    
    # 1. 学习率 (learning_rate) - 你没说具体范围，我假设一个常用的宽范围
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # 2. Batch Size
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    
    # 3. n_steps
    # 注意：PPO 要求 n_steps * n_envs > batch_size
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    
    # 4. normalize_advantage_mean (是否减去均值)
    # 对应 PPO 参数: normalize_advantage_mean
    normalize_advantage_mean = trial.suggest_categorical("normalize_advantage_mean", [True, False])
    
    # 5. ent_coef = 0 (固定值，不搜索)
    ent_coef = 0.0
    
    # 6. separate_optimizers = True (固定值，不搜索)
    separate_optimizers = True

    # ==========================================================================
    # 组装返回字典
    # ==========================================================================
    
    # 转换 gamma 和 gae_lambda
    gamma = 1 - one_minus_gamma
    gae_lambda = 1 - one_minus_gae_lambda
    
    return {
        # --- 默认参数部分 ---
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "max_grad_norm": max_grad_norm,
        
        # --- 自定义部分 ---
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        
        # --- PPO 类构造函数的特定参数 ---
        # 注意：这些参数必须在 PPO.__init__ 中存在
        "normalize_advantage_mean": normalize_advantage_mean,
        "separate_optimizers": separate_optimizers,
    }

def print_top_k_trials(algo, env_id, k=5):
    """
    寻找最新的 Optuna study pkl 文件并打印 Top K
    """
    # RL-Zoo 默认把 logs 放在 logs/算法名/report_...pkl
    log_dir = os.path.join("logs", algo)
    if not os.path.exists(log_dir):
        return

    # 找到所有 pkl 文件
    list_of_files = glob.glob(os.path.join(log_dir, "*.pkl"))
    
    if not list_of_files:
        print("\n[Info] 未找到 Optuna study 文件，跳过 Top K 打印。")
        return

    # 找到最新的文件
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"\n[Info] 加载最新的 Study 文件: {latest_file}")
    
    try:
        with open(latest_file, "rb") as f:
            study = pickle.load(f)
        
        print(f"\n========== Top {k} Trials (Sorted by Value) ==========")
        # 获取 dataframe 并排序
        df = study.trials_dataframe()
        
        # 过滤掉未完成的 trial (比如 PRUNED 或 FAIL)
        if "state" in df.columns:
            df = df[df.state == "COMPLETE"]
        
        if df.empty:
            print("没有完成的 Trial。")
            return

        # 按 value (reward) 降序排列
        top_k = df.sort_values(by="value", ascending=False).head(k)
        
        # 筛选要打印的列，避免太宽
        cols = ["number", "value"]
        # 自动添加所有 params_ 开头的列
        cols.extend([c for c in df.columns if c.startswith("params_")])
        
        # 打印
        print(top_k[cols].to_string(index=False))
        print("======================================================")
        
    except Exception as e:
        print(f"[Warning] 打印 Top K 时出错: {e}")

# ==============================================================================
# 覆盖默认的采样器
# ==============================================================================
print("--> [Custom] 已注册自定义 PPO 搜索空间 (包含 separate_optimizers)")
hyperparams_opt.HYPERPARAMS_SAMPLER["ppo"] = sample_ppo_params_custom

# ==============================================================================
# 启动训练
# ==============================================================================
if __name__ == "__main__":
    train()
    
    # 训练结束后，尝试解析参数并打印 Top K
    try:
        if "-optimize" in sys.argv or "--optimize-hyperparameters" in sys.argv:
            # 简单的参数解析
            # 查找 --algo 和 --env 参数
            algo = "ppo" # 默认为 ppo
            if "--algo" in sys.argv:
                algo = sys.argv[sys.argv.index("--algo") + 1]
            
            env = "unknown"
            if "--env" in sys.argv:
                env = sys.argv[sys.argv.index("--env") + 1]
                
            print_top_k_trials(algo, env, k=5)
    except Exception:
        pass
