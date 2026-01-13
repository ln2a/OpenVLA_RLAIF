import json
import os

MANIFEST_FILE = "/root/test/VLA/openvla/generate_preference.json"
OUTPUT_FILE = "/root/test/VLA/openvla/preference_data.json"

# 加载清单
with open(MANIFEST_FILE, "r") as f:
    manifest = json.load(f)

# 加载已完成的任务 ID
processed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # 这里的逻辑要和主脚本中的 ID 识别保持一致
            # 假设 ID 是以 task + eps 组合的
            processed_ids.add(f"{data['task']}_{data['chosen']}_{data['rejected']}")
            processed_ids.add(f"{data['task']}_{data['rejected']}_{data['chosen']}")

# 找出缺失项
missing_items = []
for item in manifest:
    task = item['task']
    ep_a = str(item['info_a']['episode'])
    ep_b = str(item['info_b']['episode'])
    if f"{task}_{ep_a}_{ep_b}" not in processed_ids:
        missing_items.append(item)

print(f"缺失条目数量: {len(missing_items)}")
for item in missing_items:
    print(f"任务: {item['task']} | 对: Ep {item['info_a']['episode']} vs {item['info_b']['episode']}")

# 将缺失项保存到一个临时清单，方便重跑
with open("/root/test/VLA/openvla/missing_manifest.json", "w") as f:
    json.dump(missing_items, f, indent=4)