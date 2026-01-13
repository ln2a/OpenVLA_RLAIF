import os
import re
import json
import random
from collections import defaultdict

# ================= 配置区 =================
VIDEO_DIR = "rollouts/FINAL_ROLLOUTS_SFT"
OUTPUT_FILE = "generate_preference.json"
TARGET_PER_TASK = 50  # 每个任务的目标对数

# 目标配比 (理想情况)
TARGET_SF = 20  # Success-Failure
TARGET_FF = 15  # Failure-Failure
TARGET_SS = 15  # Success-Success
# ========================================

def parse_filename(filename):
    """从文件名解析信息"""
    # 假设文件名格式包含 task=... success=... episode=...
    # 例如: ...--episode=350--success=True--task=pick_up_bowl...mp4
    
    # 提取任务名
    task_match = re.search(r"task=(.*?)(\.mp4|--)", filename)
    task_name = task_match.group(1) if task_match else "unknown_task"
    
    # 提取成功状态
    success_match = re.search(r"success=(True|False)", filename)
    is_success = success_match.group(1) == "True" if success_match else False
    
    # 提取 Episode 编号
    ep_match = re.search(r"episode=(\d+)", filename)
    episode = int(ep_match.group(1)) if ep_match else -1
    
    return {
        "filename": filename,
        "task": task_name,
        "success": is_success,
        "episode": episode,
        "path": os.path.join(VIDEO_DIR, filename)
    }

def generate_pairs():
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: 目录 {VIDEO_DIR} 不存在")
        return

    # 1. 扫描并归类
    all_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
    task_buckets = defaultdict(lambda: {"success": [], "failure": []})
    
    print(f"正在扫描 {len(all_files)} 个视频文件...")
    
    for f in all_files:
        info = parse_filename(f)
        if info['success']:
            task_buckets[info['task']]["success"].append(info)
        else:
            task_buckets[info['task']]["failure"].append(info)

    manifest = []
    
    print(f"\n{'='*20} 抽样统计报告 {'='*20}")
    print(f"{'Task Name (Partial)':<40} | {'SF':<4} {'FF':<4} {'SS':<4} | {'Total':<5}")
    print("-" * 70)

    # 2. 对每个任务进行配对
    for task_name, pools in task_buckets.items():
        s_pool = pools["success"]
        f_pool = pools["failure"]
        
        # --- 步骤 A: 计算配额 ---
        num_failures = len(f_pool)
        
        # 1. SF (成-败): 最多取 failures 总数，上限 TARGET_SF
        count_sf = min(num_failures, TARGET_SF)
        
        # 2. FF (败-败): 需要至少2个失败样本。复用池子。
        # 理想需要的 FF 数量
        count_ff = TARGET_FF
        # 实际限制: 如果失败样本少于2个，无法组成FF
        if num_failures < 2:
            count_ff = 0
        
        # 3. SS (成-成): 填补剩余所有空缺
        remaining_needed = TARGET_PER_TASK - count_sf - count_ff
        count_ss = remaining_needed
        
        # 确保 SS 不会变成负数 (理论上 SFT 模型成功率高，s_pool 很大，通常不会不够)
        if count_ss > len(s_pool): 
            # 极个别情况：成功率极低，不仅 F 多，S 居然不够配 SS
            # 此时回填给 FF 或 SF (此处简化处理，假设 S 足够)
            pass

        # --- 步骤 B: 执行抽样 ---
        
        # 生成 SF 对
        for _ in range(count_sf):
            s = random.choice(s_pool)
            f = random.choice(f_pool)
            manifest.append({
                "task": task_name,
                "type": "success_failure",
                "video_a": s['path'],
                "video_b": f['path'],
                "info_a": s, "info_b": f
            })
            
        # 生成 FF 对
        for _ in range(count_ff):
            # 随机抽2个不同的
            if len(f_pool) >= 2:
                pair = random.sample(f_pool, 2)
                manifest.append({
                    "task": task_name,
                    "type": "failure_failure",
                    "video_a": pair[0]['path'],
                    "video_b": pair[1]['path'],
                    "info_a": pair[0], "info_b": pair[1]
                })
        
        # 生成 SS 对
        for _ in range(count_ss):
            if len(s_pool) >= 2:
                pair = random.sample(s_pool, 2)
                manifest.append({
                    "task": task_name,
                    "type": "success_success",
                    "video_a": pair[0]['path'],
                    "video_b": pair[1]['path'],
                    "info_a": pair[0], "info_b": pair[1]
                })

        # 打印统计
        short_name = task_name[:35] + "..." if len(task_name) > 35 else task_name
        total = count_sf + count_ff + count_ss
        print(f"{short_name:<40} | {count_sf:<4} {count_ff:<4} {count_ss:<4} | {total:<5}")

    # 3. 保存文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"\n清单生成完毕！共 {len(manifest)} 对。")
    print(f"保存路径: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_pairs()