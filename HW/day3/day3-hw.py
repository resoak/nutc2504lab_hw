import time
import requests
from pathlib import Path

BASE = "https://3090api.huannago.com"
CREATE_URL = f"{BASE}/api/v1/subtitle/tasks"
WAV_PATH = "/home/pc-49/Desktop/nutc2504_lab/nutc2504lab_hw/HW/day3/Podcast_EP14_30s.wav" # 請自行更改為測試音檔路徑
auth = ("nutc2504", "nutc2504")

out_dir = Path("./out")
out_dir.mkdir(exist_ok=True)

# 1) 建立任務
with open(WAV_PATH, "rb") as f:
    r = requests.post(CREATE_URL, files={"audio": f}, timeout=60, auth=auth)
r.raise_for_status()
task_id = r.json()["id"]
print("task_id:", task_id)
print("等待轉文字...")
txt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=TXT" 
srt_url = f"{BASE}/api/v1/subtitle/tasks/{task_id}/subtitle?type=SRT"

def wait_download(url: str, max_tries=600):  # 等下載完成
    for _ in range(max_tries):
        try:
            resp = requests.get(url, timeout=(5, 60), auth=auth)
            if resp.status_code == 200:
                return resp.text
            # 還沒好通常 404
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(2)
    return None

# 2) 等 TXT(純文字)
txt_text = wait_download(txt_url, max_tries=600)
if txt_text is None:
    raise TimeoutError("轉錄逾時or錯誤")
    
# 3) 等 SRT(有時間軸+文字)
srt_text = wait_download(srt_url, max_tries=600)

# 4) 存檔（完整）
txt_path = out_dir / f"{task_id}.txt"
txt_path.write_text(txt_text, encoding="utf-8")
print("轉錄成功:", txt_path)

if srt_text is not None:
    srt_path = out_dir / f"{task_id}.srt"
    srt_path.write_text(srt_text, encoding="utf-8")
    print("轉錄成功:", srt_path)

