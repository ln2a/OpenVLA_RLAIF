import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
import h5py
import torch
from libero.libero import benchmark

import wandb

# 将当前目录加入路径，确保能找到 experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

@dataclass
class GenerateConfig:
    # 模型相关参数
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO 环境相关参数
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # 路径与日志
    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"
    use_wandb: bool = False
    wandb_project: str = "openvla_libero_eval"
    wandb_entity: str = "YOUR_WANDB_ENTITY"
    seed: int = 7

@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "checkpoint must be provided!"
    
    # 1. 设置环境与种子
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    # 2. 加载模型与处理器
    print(f"Loading model from: {cfg.pretrained_checkpoint}")
    model = get_model(cfg)
    
    # OpenVLA 特有的归一化检查
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Unnorm key {cfg.unnorm_key} not in model stats!"
        processor = get_processor(cfg)
    else:
        processor = None

    # 3. 初始化日志
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note: run_id += f"--{cfg.run_id_note}"
    
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")

    if cfg.use_wandb:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    # 4. 获取任务集
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    resize_size = get_image_resize_size(cfg)

    # 5. 开始循环评估
    total_episodes, total_successes = 0, 0
    # for task_id in tqdm.tqdm(range(task_suite.n_tasks)):
    for task_id in tqdm.tqdm(range(7, task_suite.n_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in range(cfg.num_trials_per_task):
            print(f"\nTask: {task_description} | Episode {episode_idx+1}/{cfg.num_trials_per_task}")
            
            # --- 初始化当前回合的轨迹缓存 ---
            episode_actions = []
            episode_images = [] # 用于 HDF5 保存
            replay_images = []  # 用于视频渲染
            
            obs = env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            # 设置步数限制
            max_steps_dict = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
            max_steps = max_steps_dict.get(cfg.task_suite_name, 300)

            t, done = 0, False
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                # 记录图像
                img = get_libero_image(obs, resize_size)
                replay_images.append(img)
                episode_images.append(img.astype(np.uint8)) # 强转 uint8 节省空间

                # 构建观察字典
                observation = {
                    "full_image": img,
                    "state": np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                }

                # 模型预测
                action = get_action(cfg, model, observation, task_description, processor=processor)
                
                # --- 重要：记录原始动作 ---
                action_np = action.detach().cpu().numpy() if hasattr(action, 'detach') else np.array(action)
                episode_actions.append(action_np)

                # 执行动作转换
                action_to_exec = normalize_gripper_action(action_np, binarize=True)
                if cfg.model_family == "openvla":
                    action_to_exec = invert_gripper_action(action_to_exec)

                obs, reward, done, info = env.step(action_to_exec.tolist())
                if done: break
                t += 1

            # --- 保存 HDF5 轨迹 ---
            try:
                traj_dir = Path(cfg.local_log_dir) / "trajectories" / run_id / f"task_{task_id:02d}"
                traj_dir.mkdir(parents=True, exist_ok=True)
                traj_path = traj_dir / f"ep{episode_idx:03d}_success_{int(done)}.hdf5"
                
                with h5py.File(traj_path, "w") as f:
                    f.create_dataset("observations/images", data=np.array(episode_images), compression="gzip")
                    f.create_dataset("actions", data=np.array(episode_actions))
                    f.attrs["task_description"] = str(task_description)
                    f.attrs["success"] = int(done)
                print(f"Trajectory saved: {traj_path}")
            except Exception as e:
                print(f"Trajectory Save Failed: {e}")

            # 保存视频
            save_rollout_video(replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file)

            # 更新计数
            task_successes += int(done)
            total_successes += int(done)
            task_episodes += 1
            total_episodes += 1
            
            log_file.write(f"Task: {task_description} | Ep: {episode_idx} | Success: {done}\n")
            log_file.flush()

        print(f"Task SR: {task_successes/task_episodes:.2f}")

    log_file.close()
    print("Evaluation Complete.")

if __name__ == "__main__":
    eval_libero()