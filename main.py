from contextlib import asynccontextmanager
import gc
import json
import os
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from ultralytics import YOLO

# 1. 解決 Render 系統目錄唯讀警告
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# 檔案路徑設定
MODEL_PATH = "best.pt"
CONFIG_PATH = "model_config.json"

# 建立 static 資料夾儲存 CAM 照片
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

yolo_model = None
model_config = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
  global yolo_model, model_config

  # 🎯 【記憶體優化 1】：全域關閉 PyTorch 梯度計算 & 限制單執行緒
  torch.set_grad_enabled(False)
  torch.set_num_threads(1)

  if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
    print("YOLO 模型成功載入！")
  else:
    print("⚠️ 警告：找不到 best.pt")

  if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
      model_config = json.load(f)
    print(
        f"設定檔載入成功 (版本: {model_config.get('version', 'Unknown')})"
    )
  else:
    model_config = {"threshold_1_to_2": 28.01, "threshold_2_to_3": 22.51}

  yield
  print("🛑 伺服器關閉")


app = FastAPI(
    title="Guava Quality Assessment API",
    description="芭樂自動化分級後端服務",
    version="1.2.1",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/classify")
async def classify_guava(file: UploadFile = File(...)):
  if yolo_model is None:
    raise HTTPException(status_code=500, detail="模型未就緒")

  try:
    # 🎯 【記憶體優化 2】：讀取位元組後立即解構
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 立即釋放原始暫存檔
    del contents, nparr
    gc.collect()

    if img_raw is None:
      raise HTTPException(status_code=400, detail="無效的圖片檔案")

    # 🎯 【記憶體優化 3】：一拿到圖片立刻以最省 RAM 的方式縮放至 640px 寬
    h_raw, w_raw = img_raw.shape[:2]
    target_w = 640
    target_h = int(h_raw * (target_w / float(w_raw)))

    img_resized = cv2.resize(
        img_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    del img_raw  # 釋放高解析度原圖
    gc.collect()

    h, w, _ = img_resized.shape

    # 🎯 【記憶體優化 4】：YOLO 輕量推論設定
    results = yolo_model(
        img_resized,
        imgsz=640,
        conf=0.45,
        max_det=1,
        verbose=False,
        half=False,  # CPU 模式下關閉 FP16 避免異常複製
    )

    detected = False
    crop_img = None
    annotated_img = img_resized.copy()

    if len(results) > 0 and len(results[0].boxes) > 0:
      box = results[0].boxes[0]
      x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
      conf = float(box.conf[0])

      # 畫綠框
      cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
      cv2.putText(
          annotated_img,
          f"Guava {conf:.2f}",
          (x1, max(25, y1 - 10)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 255, 0),
          2,
      )

      # 計算中心點與 250x250 ROI
      cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
      ymin, ymax = max(0, cy - 125), min(h, cy + 125)
      xmin, xmax = max(0, cx - 125), min(w, cx + 125)

      crop_img = img_resized[ymin:ymax, xmin:xmax].copy()
      cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
      detected = True

    # 釋放 YOLO 內部推論矩陣
    del results
    gc.collect()

    # 備援機制 (未偵測到時)
    if not detected:
      cv2.putText(
          annotated_img,
          "NO GUAVA DETECTED (Fallback Center)",
          (20, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          (0, 0, 255),
          2,
      )
      ymin, ymax = max(0, int(h / 2) - 125), min(h, int(h / 2) + 125)
      xmin, xmax = max(0, int(w / 2) - 125), min(w, int(w / 2) + 125)
      cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
      crop_img = img_resized[ymin:ymax, xmin:xmax].copy()

    if crop_img.shape[0] != 250 or crop_img.shape[1] != 250:
      crop_img = cv2.resize(crop_img, (250, 250))

    # 儲存標註照片
    save_filename = "latest_guava.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_filename)
    cv2.imwrite(save_path, annotated_img)

    del img_resized, annotated_img
    gc.collect()

    image_public_url = f"https://guava-backend-ekg3.onrender.com/static/uploads/{save_filename}"

    # 計算 K-Means 特徵 (灰階標準差)
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    del crop_img
    fruit_mask = gray > 15

    if np.sum(fruit_mask) > 0:
      score = float(np.std(gray[fruit_mask]))
    else:
      score = float(np.std(gray))

    del gray, fruit_mask
    gc.collect()

    # 門檻與分級
    t12 = model_config.get("threshold_1_to_2", 28.01)
    t23 = model_config.get("threshold_2_to_3", 22.51)

    if score >= t12:
      group = 1
      quality = "Excellent"
      desc = "Fruit skin texture is bumpy and full"
    elif score >= t23:
      group = 2
      quality = "Good"
      desc = "Fruit skin texture is medium"
    else:
      group = 3
      quality = "Smooth"
      desc = "Fruit skin texture is relatively smooth"

    return {
        "status": "success",
        "data": {
            "filename": file.filename,
            "score": round(score, 2),
            "group": group,
            "quality": quality,
            "description": desc,
            "yolo_detected": detected,
            "image_url": image_public_url,
            "version": model_config.get("version", "Unknown"),
        },
    }

  except Exception as e:
    gc.collect()
    raise HTTPException(
        status_code=500, detail=f"伺服器處理失敗: {str(e)}"
    )


@app.get("/")
def health_check():
  return {
      "status": "online",
      "message": "芭樂 AI 分級後端服務運作中 (20260929-LowRAM)",
      "current_config_version": model_config.get("version", "Unknown"),
  }
