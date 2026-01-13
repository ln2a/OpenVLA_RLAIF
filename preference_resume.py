import os
import json
import time
import base64
import cv2
from openai import OpenAI
from pathlib import Path

# ================= 配置区 =================
# 1. 使用你 clean 过的清单
MANIFEST_FILE = "/root/test/VLA/openvla/generate_preference_clean.json"
# 2. 结果保存到去重后的文件，并在其基础上追加
OUTPUT_FILE = "/root/test/VLA/openvla/preference_data_unique.json"

API_KEY = "mykey" 
BASE_URL = "https://api.chataiapi.com/v1"
MODEL_NAME = "gpt-4o"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 核心工具函数 =================

def load_processed_ids(output_file):
    """加载已完成的任务ID，用于断点续传"""
    processed = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task = data['task']
                    # 排序 ID 确保 A-B 和 B-A 被视为同一个
                    pair = tuple(sorted([str(data['chosen']), str(data['rejected'])]))
                    processed.add((task, pair))
                except: continue
    return processed

def encode_video_frames(video_path, num_frames=10):
    """从视频抽取关键帧并转 Base64"""
    if not os.path.exists(video_path):
        # 简单容错：如果路径不对，尝试加前缀或打印错误
        print(f"  !! 错误：视频文件不存在: {video_path}")
        raise ValueError(f"File not found: {video_path}")

    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        video.release()
        raise ValueError(f"无法读取视频: {video_path}")

    interval = max(1, total_frames // num_frames)
    frames_base64 = []
    
    curr_frame = 0
    while video.isOpened() and len(frames_base64) < num_frames:
        success, frame = video.read()
        if not success: break
        
        if curr_frame % interval == 0:
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64_str = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append(b64_str)
        curr_frame += 1
        
    video.release()
    return frames_base64

def get_gpt4o_verdict(item):
    """调用 GPT-4o 判断 S-S 或 F-F 对 (含重试机制)"""
    task_desc = item['task']
    pair_type = item['type']
    path_a = item['video_a']
    path_b = item['video_b']

    try:
        frames_a = encode_video_frames(path_a)
        frames_b = encode_video_frames(path_b)
    except Exception as e:
        print(f"  !! 视频读取错误: {e}")
        return None

    if pair_type == "success_success":
        criteria = (
            "Both attempts SUCCEEDED. Pick the better one based on:\n"
            "1. Efficiency: Fewer steps, no redundant motions.\n"
            "2. Safety: No sliding or collisions.\n"
            "3. Precision: More stable grasp/placement."
        )
    else: # failure_failure
        criteria = (
            "Both attempts FAILED. Pick the one with better potential:\n"
            "1. Intent: Correct object identification.\n"
            "2. Progress: Reached a further stage (Reach > Grasp > Lift).\n"
            "3. Logic: Smoother trajectory."
        )

    system_prompt = (
        f"You are a robotic evaluation expert. Task: {task_desc}\n"
        f"{criteria}\n"
        "Compare Video A and Video B. "
        "Output JSON ONLY: {\"winner\": \"A\" or \"B\", \"reason\": \"One short English sentence\"}"
    )

    content = [{"type": "text", "text": "Analyze these two videos."}]
    content.append({"type": "text", "text": "--- Video A ---"})
    for f in frames_a:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}})
    content.append({"type": "text", "text": "--- Video B ---"})
    for f in frames_b:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}})

    max_retries = 3
    timeout_sec = 45.0 

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=timeout_sec
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"\n  请求失败 (第 {attempt + 1}/{max_retries} 次): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None

# ================= 主循环 =================

def run():
    print(f"正在读取清单: {MANIFEST_FILE}")
    if not os.path.exists(MANIFEST_FILE):
        print("错误：找不到清单文件！")
        return

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    # 1. 加载已有进度
    processed_set = load_processed_ids(OUTPUT_FILE)
    print(f"已完成唯一条目: {len(processed_set)}")
    print(f"目标总条目: {len(manifest)}")
    
    # 2. 追加模式打开 (append)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
        
        success_count = 0
        
        for i, item in enumerate(manifest):
            task_name = item['task']
            ep_a = str(item['info_a']['episode'])
            ep_b = str(item['info_b']['episode'])
            pair_type = item['type']

            # 3. 检查跳过逻辑
            current_key = (task_name, tuple(sorted([ep_a, ep_b])))
            if current_key in processed_set:
                # 如果你想看跳过了哪些，可以取消下面这行的注释
                # print(f"[{i+1}/{len(manifest)}] 跳过已存在的对: {ep_a} vs {ep_b}")
                continue

            print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b}", end="", flush=True)

            result_entry = None

            # 策略 1: 成功-失败对 (规则判决)
            if pair_type == "success_failure":
                print(" -> [Auto Rule]", flush=True)
                is_a_success = item['info_a']['success']
                winner = ep_a if is_a_success else ep_b
                loser = ep_b if is_a_success else ep_a
                
                result_entry = {
                    "task": task_name,
                    "pair_type": "success_failure",
                    "chosen": winner,
                    "rejected": loser,
                    "reason": "Deterministic: The chosen trajectory succeeded while the rejected one failed.",
                    "model": "rule_based"
                }

            # 策略 2: 同类对比 (API 判决)
            else:
                print(" -> [GPT-4o] Processing...", end="", flush=True)
                verdict = get_gpt4o_verdict(item)
                
                print("\r" + " " * 80 + "\r", end="")
                
                if verdict:
                    print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b} -> [GPT-4o Done]", flush=True)
                    winner_label = verdict.get("winner", "A").upper()
                    
                    if winner_label == "A":
                        chosen, rejected = ep_a, ep_b
                    else:
                        chosen, rejected = ep_b, ep_a
                    
                    result_entry = {
                        "task": task_name,
                        "pair_type": pair_type,
                        "chosen": chosen,
                        "rejected": rejected,
                        "reason": verdict.get("reason", "N/A"),
                        "model": MODEL_NAME
                    }
                    time.sleep(1)
                else:
                    print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b} -> [Skip] API Failed")

            # 写入结果
            if result_entry:
                print(f"    Winner: {result_entry['chosen']} | Reason: {result_entry['reason']}")
                print(f"    {'-'*50}")
                f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                f_out.flush() # 立即保存
                success_count += 1

    print(f"\n{'='*30}")
    print(f"补齐完成！本次新增写入: {success_count} 条")
    print(f"最终文件位置: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()