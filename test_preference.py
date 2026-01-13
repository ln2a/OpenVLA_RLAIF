import os
import re
import time
import random
import json
import base64
import cv2  # 需要 pip install opencv-python
from pathlib import Path
from openai import OpenAI  # 需要 pip install openai

# ================= 配置区 =================
# 务必确认你的 Key 支持 GPT-4o
API_KEY = "sk-2gBVDbmAD3Uux4H2APxBn28lABYJfe8zfpc7ESohIDGSKEVv" 
BASE_URL = "https://api.chataiapi.com/v1"  # OpenAI 兼容接口通常需要加 /v1
VIDEO_DIR = "rollouts/FINAL_ROLLOUTS_SFT"
TARGET_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_r"
OUTPUT_FILE = "preference_data_task00.jsonl"
MODEL_NAME = "gpt-4o"

# 初始化 OpenAI Client
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ================= 工具函数 =================
def extract_info(filename):
    """从文件名提取Episode号、成功状态"""
    ep_match = re.search(r"episode=(\d+)", filename)
    success_match = re.search(r"success=(True|False)", filename)
    return {
        "ep": ep_match.group(1) if ep_match else "unk",
        "success": success_match.group(1) == "True" if success_match else False,
        "filename": filename
    }

def encode_video_frames(video_path, num_frames=10):
    """
    使用 OpenCV 从视频中均匀抽取关键帧并转为 Base64
    GPT-4o 建议输入 10-20 帧即可理解动作
    """
    video = cv2.VideoCapture(str(video_path))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果视频无法打开
    if total_frames == 0:
        video.release()
        raise ValueError(f"无法读取视频: {video_path}")

    # 计算采样间隔
    interval = max(1, total_frames // num_frames)
    frames_base64 = []
    
    curr_frame = 0
    while video.isOpened() and len(frames_base64) < num_frames:
        success, frame = video.read()
        if not success:
            break
            
        # 按照间隔采样
        if curr_frame % interval == 0:
            # 压缩图片以节省 Token (转为 JPEG, 质量 70)
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            b64_str = base64.b64encode(buffer).decode("utf-8")
            frames_base64.append(b64_str)
            
        curr_frame += 1
        
    video.release()
    return frames_base64

def call_gpt4o_preference(vid_a, vid_b, is_success_pair):
    """构建 GPT-4o 的多模态 Payload"""
    video_path = Path(VIDEO_DIR)
    path_a = video_path / vid_a['filename']
    path_b = video_path / vid_b['filename']

    # 1. 抽取帧
    frames_a = encode_video_frames(path_a, num_frames=10)
    frames_b = encode_video_frames(path_b, num_frames=10)

    # 2. 构建 Prompt (保持你的极简风格)
    if is_success_pair:
        criteria = (
            "Both succeeded. Criteria:\n"
            "1. Efficiency: No redundant/post-task motions.\n"
            "2. Safety: No sliding or non-target collisions.\n"
            "3. Precision: Centered and stable grasp/place points."
        )
    else:
        criteria = (
            "Both failed. Criteria:\n"
            "1. Intent: Correct target identification.\n"
            "2. Progress: Further task stage (Reach > Grasp > Lift).\n"
            "3. Logic: Purposeful and smooth trajectory."
        )

    system_prompt = (
        f"You are a robotic evaluation expert. Task: {TARGET_TASK}\n"
        f"{criteria}\n"
        "Output strictly in JSON format: {\"winner\": \"A\" or \"B\", \"reason\": \"One short English sentence\"}"
    )

    # 3. 构建消息体 (交替插入文本和图片)
    # 结构：[Prompt -> Video A Frames -> Separator -> Video B Frames]
    messages_content = [{"type": "text", "text": "Analyze these two robot execution videos."}]
    
    # 添加视频 A
    messages_content.append({"type": "text", "text": "--- Video A Frames ---"})
    for f in frames_a:
        messages_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"} # detail=low 省钱且速度快
        })
        
    # 添加视频 B
    messages_content.append({"type": "text", "text": "--- Video B Frames ---"})
    for f in frames_b:
        messages_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{f}", "detail": "low"}
        })
        
    # 发送请求
    print(f"DEBUG: 正在请求 GPT-4o (Payload包含 {len(frames_a)+len(frames_b)} 张图片)...", flush=True)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": messages_content}
            ],
            response_format={"type": "json_object"}, # 强制 JSON 模式 (GPT-4o 特性)
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI API Error: {str(e)}")

# ================= 主程序 =================
def run_annotation():
    print(f"DEBUG: 正在扫描文件夹: {VIDEO_DIR}", flush=True)
    
    if not os.path.exists(VIDEO_DIR):
        print(f"错误: 找不到文件夹 {VIDEO_DIR}")
        return

    # 1. 搜集并筛选视频
    all_files = [f for f in os.listdir(VIDEO_DIR) if TARGET_TASK in f and f.endswith(".mp4")]
    parsed_vids = [extract_info(f) for f in all_files]
    
    success_pool = [v for v in parsed_vids if v['success']]
    failure_pool = [v for v in parsed_vids if not v['success']]
    
    print(f"池化完成: 成功 {len(success_pool)} / 失败 {len(failure_pool)}", flush=True)

    # 2. 抽样逻辑 (此处保留你的测试逻辑：3对成功 + 2对失败)
    final_pairs = []
    if len(success_pool) >= 2:
        s = random.sample(success_pool, min(len(success_pool), 6))
        for i in range(0, len(s)-1, 2):
            final_pairs.append((s[i], s[i+1], True))
    
    if len(failure_pool) >= 2:
        f = random.sample(failure_pool, min(len(failure_pool), 4))
        for i in range(0, len(f)-1, 2):
            final_pairs.append((f[i], f[i+1], False))

    if not final_pairs:
        print("DEBUG: 未找到足够的视频对。", flush=True)
        return

    print(f"\n{'='*20} 开始执行 GPT-4o 标注 {'='*20}", flush=True)

    # 3. 循环标注并保存 (使用 "w" 模式覆盖)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for i, (vid_a, vid_b, is_success) in enumerate(final_pairs):
            try:
                print(f"正在处理第 {i+1}/{len(final_pairs)} 对: Ep{vid_a['ep']} vs Ep{vid_b['ep']}", flush=True)
                
                raw_res = call_gpt4o_preference(vid_a, vid_b, is_success)
                
                # 清洗 JSON 结果 (虽然 response_format 保证了 JSON，但防止 markdown 包裹)
                clean_json = raw_res.strip().replace("```json", "").replace("```", "")
                data = json.loads(clean_json)
                
                winner_ep = vid_a['ep'] if data['winner'].upper() == 'A' else vid_b['ep']
                loser_ep = vid_b['ep'] if data['winner'].upper() == 'A' else vid_a['ep']

                print(f"  -> 结果: Winner Ep {winner_ep} > Loser Ep {loser_ep}")
                print(f"  -> 理由: {data.get('reason', 'N/A')}")

                result = {
                    "task": TARGET_TASK,
                    "pair_type": "success" if is_success else "failure",
                    "chosen": winner_ep,
                    "rejected": loser_ep,
                    "reason": data.get('reason', ''),
                    "model": MODEL_NAME,
                    "timestamp": time.time()
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

                # 避免并发限制，稍微休眠
                time.sleep(1)

            except Exception as e:
                print(f"  !! 处理第 {i+1} 对时出错: {str(e)}", flush=True)

if __name__ == "__main__":
    run_annotation()