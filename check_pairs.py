import json
from collections import defaultdict, Counter

FILE_PATH = "/root/test/VLA/openvla/generate_preference_clean.json"

def validate():
    if not os.path.exists(FILE_PATH):
        print(f"错误：找不到文件 {FILE_PATH}")
        return

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"总条目数: {len(data)}")

    # 1. 唯一性校验
    unique_keys = set()
    duplicates = []
    
    # 2. 比例统计
    # stats[task] = {"success_failure": 0, "failure_failure": 0, "success_success": 0}
    stats = defaultdict(lambda: Counter())

    for item in data:
        task = item['task']
        # 生成唯一标识，不分 A/B 顺序
        pair = tuple(sorted([str(item['info_a']['episode']), str(item['info_b']['episode'])]))
        key = (task, pair)

        if key in unique_keys:
            duplicates.append(key)
        else:
            unique_keys.add(key)
        
        # 统计类型
        stats[task][item['type']] += 1

    # --- 输出结果 ---
    print("\n" + "="*50)
    print("1. 唯一性检查结果:")
    if not duplicates:
        print("   ✅ 所有偏好对均唯一，无重复！")
    else:
        print(f"   ❌ 发现 {len(duplicates)} 组重复对：")
        for d in duplicates[:5]: print(f"      - {d}")

    print("\n" + "="*50)
    print("2. 任务配比检查 (目标 SF:FF:SS = 20:15:15):")
    print(f"{'Task Name (Partial)':<40} | {'SF':<3} : {'FF':<3} : {'SS':<3} | {'Total'}")
    print("-" * 70)

    for task, counts in stats.items():
        sf = counts.get("success_failure", 0)
        ff = counts.get("failure_failure", 0)
        ss = counts.get("success_success", 0)
        total = sf + ff + ss
        
        short_name = task[:37] + "..." if len(task) > 37 else task
        print(f"{short_name:<40} | {sf:<3} : {ff:<3} : {ss:<3} | {total}")

if __name__ == "__main__":
    import os
    validate()