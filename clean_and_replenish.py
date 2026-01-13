import os
import re
import json
import random
from collections import defaultdict, Counter

# ================= 配置区 =================
# 输入/输出文件
ORIGINAL_FILE = "generate_preference.json"
OUTPUT_FILE = "generate_preference_clean.json"
VIDEO_DIR = "rollouts/FINAL_ROLLOUTS_SFT"

# 目标设定
TOTAL_TARGET = 500
TARGET_PER_TASK = 50 
# 配比目标
TARGET_SF = 20
TARGET_FF = 15
TARGET_SS = 15
# ========================================

def parse_filename(filename):
    """从文件名解析信息 (保持原逻辑)"""
    task_match = re.search(r"task=(.*?)(\.mp4|--)", filename)
    task_name = task_match.group(1) if task_match else "unknown_task"
    
    success_match = re.search(r"success=(True|False)", filename)
    is_success = success_match.group(1) == "True" if success_match else False
    
    ep_match = re.search(r"episode=(\d+)", filename)
    episode = int(ep_match.group(1)) if ep_match else -1
    
    return {
        "filename": filename,
        "task": task_name,
        "success": is_success,
        "episode": episode,
        "path": os.path.join(VIDEO_DIR, filename)
    }

def get_pair_key(item):
    """生成唯一键，用于去重 (任务名 + 排序后的Episode对)"""
    ep_a = str(item['info_a']['episode'])
    ep_b = str(item['info_b']['episode'])
    # 排序确保 A vs B 和 B vs A 被视为同一个
    pair = tuple(sorted([ep_a, ep_b]))
    return (item['task'], pair)

def run_clean_and_replenish():
    # --- 步骤 1: 读取并清洗旧数据 ---
    print(f"正在读取 {ORIGINAL_FILE} ...")
    if not os.path.exists(ORIGINAL_FILE):
        print("错误：找不到原始清单文件。")
        return

    with open(ORIGINAL_FILE, "r") as f:
        original_data = json.load(f)

    clean_manifest = []
    existing_keys = set()
    
    # 统计当前每个任务已有的类型数量
    task_stats = defaultdict(lambda: {"total": 0, "SF": 0, "FF": 0, "SS": 0})

    print(f"原始条目数: {len(original_data)}")
    
    for item in original_data:
        key = get_pair_key(item)
        if key not in existing_keys:
            existing_keys.add(key)
            clean_manifest.append(item)
            
            # 统计分布
            t_name = item['task']
            p_type = item['type'] # success_failure, failure_failure, etc.
            task_stats[t_name]["total"] += 1
            if p_type == "success_failure": task_stats[t_name]["SF"] += 1
            elif p_type == "failure_failure": task_stats[t_name]["FF"] += 1
            elif p_type == "success_success": task_stats[t_name]["SS"] += 1

    print(f"去重后条目数: {len(clean_manifest)}")
    
    if len(clean_manifest) >= TOTAL_TARGET:
        print("注意：去重后数据量已满足或超过 500 条，无需补充。")
        # 即使超过，也建议保存一份 clean 版
    else:
        to_add = TOTAL_TARGET - len(clean_manifest)
        print(f"需要补充: {to_add} 条")

        # --- 步骤 2: 扫描视频库 (获取素材池) ---
        if not os.path.exists(VIDEO_DIR):
            print(f"Error: 目录 {VIDEO_DIR} 不存在，无法进行补充。")
            return

        all_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")]
        task_buckets = defaultdict(lambda: {"success": [], "failure": []})
        
        print(f"正在扫描视频库 ({len(all_files)} 个文件)...")
        for f in all_files:
            info = parse_filename(f)
            if info['success']:
                task_buckets[info['task']]["success"].append(info)
            else:
                task_buckets[info['task']]["failure"].append(info)

        # --- 步骤 3: 补充缺失数据 ---
        # 我们遍历每个任务，检查它是否缺数据，缺什么类型的数据
        
        for task_name, pools in task_buckets.items():
            current_count = task_stats[task_name]["total"]
            if current_count >= TARGET_PER_TASK:
                continue # 该任务已满
            
            needed_for_task = TARGET_PER_TASK - current_count
            print(f"任务 {task_name[:20]}... 当前 {current_count}，需补 {needed_for_task} 条")
            
            s_pool = pools["success"]
            f_pool = pools["failure"]
            
            # 尝试补充 needed_for_task 次
            # 这里的逻辑：优先补 SF，再补 FF，最后补 SS，尽量贴近目标比例
            
            for _ in range(needed_for_task):
                stats = task_stats[task_name]
                
                # 决策：下一个补什么类型？
                # 1. 优先补 SF (目标 20)
                if stats["SF"] < TARGET_SF and len(s_pool) > 0 and len(f_pool) > 0:
                    try_type = "success_failure"
                # 2. 其次补 FF (目标 15)
                elif stats["FF"] < TARGET_FF and len(f_pool) >= 2:
                    try_type = "failure_failure"
                # 3. 最后补 SS (填满余下)
                elif len(s_pool) >= 2:
                    try_type = "success_success"
                else:
                    # 实在没素材了（比如全是失败且失败样本很少），跳过
                    continue

                # 尝试生成一个不重复的对
                new_item = None
                max_retries = 50 # 防止死循环
                
                for attempt in range(max_retries):
                    if try_type == "success_failure":
                        video_a = random.choice(s_pool)
                        video_b = random.choice(f_pool)
                        info_a, info_b = video_a, video_b
                    elif try_type == "failure_failure":
                        pair = random.sample(f_pool, 2)
                        video_a, video_b = pair[0], pair[1]
                        info_a, info_b = pair[0], pair[1]
                    elif try_type == "success_success":
                        pair = random.sample(s_pool, 2)
                        video_a, video_b = pair[0], pair[1]
                        info_a, info_b = pair[0], pair[1]
                    
                    # 检查是否重复
                    temp_item = {
                        "task": task_name, "type": try_type,
                        "info_a": info_a, "info_b": info_b
                    }
                    temp_key = get_pair_key(temp_item)
                    
                    if temp_key not in existing_keys:
                        # 找到了唯一的！构建完整对象
                        new_item = {
                            "task": task_name,
                            "type": try_type,
                            "video_a": info_a['path'],
                            "video_b": info_b['path'],
                            "info_a": info_a,
                            "info_b": info_b
                        }
                        existing_keys.add(temp_key) # 立即加入防止本轮内部重复
                        break
                
                if new_item:
                    clean_manifest.append(new_item)
                    # 更新统计
                    task_stats[task_name]["total"] += 1
                    if try_type == "success_failure": task_stats[task_name]["SF"] += 1
                    elif try_type == "failure_failure": task_stats[task_name]["FF"] += 1
                    elif try_type == "success_success": task_stats[task_name]["SS"] += 1
                else:
                    print(f"  警告: 无法为 {task_name} 生成唯一的 {try_type} (尝试 {max_retries} 次失败)")

    # --- 步骤 4: 保存结果 ---
    print(f"\n最终总数: {len(clean_manifest)}")
    
    # 最后简单的完整性检查
    final_keys = [get_pair_key(i) for i in clean_manifest]
    if len(final_keys) != len(set(final_keys)):
        print("严重警告：最终结果中仍检测到重复项！请检查代码逻辑。")
    else:
        print("完整性检查通过：所有轨迹对均为唯一。")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_manifest, f, indent=2, ensure_ascii=False)
    
    print(f"已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_clean_and_replenish()