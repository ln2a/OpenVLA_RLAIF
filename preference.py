import os
import json
import time
import base64
import cv2
from openai import OpenAI
from pathlib import Path

# ================= 配置区 =================
MANIFEST_FILE = "/root/test/VLA/openvla/generate_preference.json"
OUTPUT_FILE = "/root/test/VLA/openvla/preference_data.json"

API_KEY = "mykey" 
BASE_URL = "https://api.chataiapi.com/v1"
MODEL_NAME = "gpt-4o"

# 初始化客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= 核心工具函数 =================

def encode_video_frames(video_path, num_frames=10):
    """从视频抽取关键帧并转 Base64"""
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

    # 1. 准备图片 (如果读取失败直接返回，不重试)
    try:
        frames_a = encode_video_frames(path_a)
        frames_b = encode_video_frames(path_b)
    except Exception as e:
        print(f"  !! 视频读取错误: {e}")
        return None

    # 2. 构建 Prompt
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

    # 3. 发送请求 (带重试和超时逻辑)
    max_retries = 3
    timeout_sec = 30.0 # 分半超时

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
                timeout=timeout_sec  # <--- 设置超时时间
            )
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"\n  请求失败 (第 {attempt + 1}/{max_retries} 次): {str(e)}")
            if attempt < max_retries - 1:
                print(f"  等待 5 秒后重试...")
                time.sleep(5)
            else:
                print(f"  达到最大重试次数，跳过此条数据。")
                return None

# ================= 主循环 =================

def run():
    print(f"正在读取清单: {MANIFEST_FILE}")
    if not os.path.exists(MANIFEST_FILE):
        print("错误：找不到清单文件！请先运行 generate_manifest.py")
        return

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)

    print(f"共加载 {len(manifest)} 个待处理对。")
    print(f"结果将保存至: {OUTPUT_FILE}")

    # 初始化清空文件 (注意：这会覆盖旧文件，如果需要断点续传请去掉这块或改为手动检查)
    with open(OUTPUT_FILE, "w") as f:
        pass

    success_count = 0
    
    for i, item in enumerate(manifest):
        pair_type = item['type']
        task_name = item['task']
        ep_a = str(item['info_a']['episode'])
        ep_b = str(item['info_b']['episode'])
        
        result_entry = None
        
        # 打印当前进度头（不换行）
        print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b}", end="", flush=True)

        # --- 策略分支 ---
        
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
            
            # 清除 Waiting 提示
            print("\r" + " " * 80 + "\r", end="")
            
            if verdict:
                print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b} -> [GPT-4o Done]", flush=True)
                winner_label = verdict.get("winner", "A").upper()
                reason_text = verdict.get("reason", "N/A")
                
                if winner_label == "A":
                    chosen, rejected = ep_a, ep_b
                else:
                    chosen, rejected = ep_b, ep_a
                
                result_entry = {
                    "task": task_name,
                    "pair_type": pair_type,
                    "chosen": chosen,
                    "rejected": rejected,
                    "reason": reason_text,
                    "model": MODEL_NAME
                }
                time.sleep(1) # 避免并发限制
            else:
                print(f"[{i+1}/{len(manifest)}] {pair_type} | Ep {ep_a} vs {ep_b} -> [Skip] API Failed 3 times", flush=True)

        # --- 打印结果并写入文件 ---
        if result_entry:
            # 1. 终端详细打印
            print(f"    Winner: Ep {result_entry['chosen']}  (Loser: Ep {result_entry['rejected']})")
            print(f"    Reason: {result_entry['reason']}")
            print(f"    {'-'*50}")

            # 2. 追加写入文件
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
                f.flush() # 强制立即写入磁盘
            success_count += 1

    print(f"\n{'='*30}")
    print(f"标注完成！")
    print(f"成功写入: {success_count} / {len(manifest)}")
    print(f"文件位置: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()