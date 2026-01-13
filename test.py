import json
import os
from collections import defaultdict

# 你的目标文件路径
FILE_PATH = "/root/test/VLA/openvla/preference_data_unique.json"

def check_data():
    if not os.path.exists(FILE_PATH):
        print(f"错误：文件不存在 -> {FILE_PATH}")
        return

    print(f"正在读取文件: {FILE_PATH} ...\n")
    
    # 用于查重的集合: 存储 (task, 排序后的episode对)
    seen_pairs = set()
    duplicates = []
    
    # 统计数据: stats[task_name][pair_type] = count
    stats = defaultdict(lambda: defaultdict(int))
    
    total_lines = 0
    
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告: 第 {line_num} 行无法解析为JSON")
                continue
            
            total_lines += 1
            
            # 提取关键字段
            task = data.get('task', 'Unknown_Task')
            pair_type = data.get('pair_type', 'unknown')
            
            # 获取 Episode ID (兼容字符串或整数)
            chosen = str(data.get('chosen', ''))
            rejected = str(data.get('rejected', ''))
            
            # --- 查重逻辑 ---
            # 将 (chosen, rejected) 排序，确保 (1, 2) 和 (2, 1) 被视为同一对
            pair_key = (task, tuple(sorted([chosen, rejected])))
            
            if pair_key in seen_pairs:
                duplicates.append(f"行 {line_num} 重复: Task={task[:20]}... | Ep {chosen} vs {rejected}")
            else:
                seen_pairs.add(pair_key)
            
            # --- 统计逻辑 ---
            stats[task][pair_type] += 1

    # ================= 输出报告 =================
    print(f"{'='*20} 数据质量检查报告 {'='*20}")
    print(f"原始总行数: {total_lines}")
    print(f"唯一有效对: {len(seen_pairs)}")
    
    print(f"\n[1] 重复性检查结果:")
    if not duplicates:
        print("   ✅ 完美！文件中没有任何重复的偏好对。")
    else:
        print(f"   ❌ 警告：发现 {len(duplicates)} 个重复项！")
        for d in duplicates[:5]:
            print(f"      {d}")
        if len(duplicates) > 5:
            print(f"      ... (还有 {len(duplicates)-5} 项)")

    print(f"\n[2] 任务分布统计表 (SF=成功vs失败, FF=失败vs失败, SS=成功vs成功):")
    print("-" * 95)
    print(f"{'Task Name':<55} | {'SF':<4} {'FF':<4} {'SS':<4} | {'Sum':<5}")
    print("-" * 95)
    
    # 按任务名排序输出
    sorted_tasks = sorted(stats.keys())
    
    grand_total = 0
    for task in sorted_tasks:
        counts = stats[task]
        sf = counts.get('success_failure', 0)
        ff = counts.get('failure_failure', 0)
        ss = counts.get('success_success', 0)
        # 兼容人工手动补全的类型标记
        manual = counts.get('manual_check', 0) + counts.get('manual_recovery', 0)
        
        row_total = sf + ff + ss + manual
        grand_total += row_total
        
        # 缩短超长任务名以便打印
        display_name = (task[:52] + '..') if len(task) > 52 else task
        
        extra_info = ""
        if manual > 0: extra_info = f"(+Manual:{manual})"
            
        print(f"{display_name:<55} | {sf:<4} {ff:<4} {ss:<4} | {row_total:<5} {extra_info}")

    print("-" * 95)
    print(f"总计 (Grand Total): {grand_total}")
    
    # 简单校验
    if grand_total == 500:
        print("\n✅ 数据集完整性校验通过：总数为 500 条。")
    else:
        print(f"\n⚠️ 数据集数量提示：当前总数为 {grand_total} (目标 500)，差 {500 - grand_total} 条。")

if __name__ == "__main__":
    check_data()