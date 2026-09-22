from contextlib import asynccontextmanager
import gc
import json
import os
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # 👈 1. 引入 StaticFiles
import numpy as np
import torch
from ultralytics import YOLO

# 1. 全域變數與檔案路徑設定
MODEL_PATH = "best.pt"
CONFIG_PATH = "model_config.json"

# 🎯 建立 static 資料夾用於儲存 ESP32-CAM 上傳的照片
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

yolo_model = None
model_config = {}


# 2. 使用 lifespan 管理伺服器生命週期
@asynccontextmanager
async def lifespan(app: FastAPI):
  global yolo_model, model_config

  # 強制 PyTorch 只使用單執行緒，避免免費版單核 CPU 爭搶資源
  torch.set_num_threads(1)

  if os.path.exists(MODEL_PATH):
    yolo_model = YOLO(MODEL_PATH)
    print("✅ YOLO 模型成功載入！")
  else:
    print("⚠️ 警告：找不到 best.pt")

  if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
      model_config = json.load(f)
    print(
        f"✅ 設定檔載入成功 (版本: {model_config.get('version', 'Unknown')})"
    )
  else:
    model_config = {"threshold_1_to_2": 15.0, "threshold_2_to_3": 9.0}

  yield
  print("🛑 伺服器關閉")


# 3. 初始化 FastAPI
app = FastAPI(
    title="Guava Quality Assessment API",
    description="芭樂自動化分級後端服務 (極速防爆版)",
    version="1.2.0",
    lifespan=lifespan,
)

# 👈 2. 掛載 /static 路徑，讓外部瀏覽器可以直接存取照片
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 4. 核心檢測 API 路由
@app.post("/api/v1/classify")
async def classify_guava(file: UploadFile = File(...)):
  if yolo_model is None:
    raise HTTPException(status_code=500, detail="模型未就緒")

  try:
    # A. 讀取圖片 Bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 釋放原始 Buffer
    del contents, nparr
    gc.collect()

    if img is None:
      raise HTTPException(status_code=400, detail="無效的圖片檔案")

    # B. 【高速+防爆】：限制最高解析度
    h_orig, w_orig = img.shape[:2]
    if max(h_orig, w_orig) > 1200:
      scale_down = 1200 / float(max(h_orig, w_orig))
      img = cv2.resize(
          img,
          (int(w_orig * scale_down), int(h_orig * scale_down)),
          interpolation=cv2.INTER_AREA,
      )

    # C. 縮放到標準 800px 寬度
    target_w = 800
    scale = target_w / img.shape[1]
    target_h = int(img.shape[0] * scale)
    img_resized = cv2.resize(img, (target_w, target_h))

    del img
    gc.collect()

    h, w, _ = img_resized.shape

    # D. 【高速推論核心】：YOLO 偵測
    with torch.no_grad():
      results = yolo_model(img_resized, imgsz=640, verbose=False)

    detected = False
    crop_img = None

    # 複製一份影像用於繪製網頁顯示的標註圖 (Annotated Image)
    annotated_img = img_resized.copy()

    for r in results:
      for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
        conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0

        # 🎯 畫上 YOLO 綠色框 (Bounding Box) 與 信心度文字
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label_text = f"Guava {conf:.2f}"
        cv2.putText(
            annotated_img,
            label_text,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # 裁切 ROI (250x250) 供後續 K-Means 質地計算
        ymin, ymax = max(0, cy - 125), min(h, cy + 125)
        xmin, xmax = max(0, cx - 125), min(w, cx + 125)
        crop_img = img_resized[ymin:ymax, xmin:xmax]

        # 畫上紅框代表實際採樣分析的 ROI 區域
        cv2.rectangle(
            annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2
        )

        if crop_img.shape[0] != 250 or crop_img.shape[1] != 250:
          crop_img = cv2.resize(crop_img, (250, 250))

        detected = True
        break
      if detected:
        break

    if not detected:
      # 沒偵測到目標時：在圖片上方標示紅色警告文字，並抓中央區域
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
      crop_img = img_resized[ymin:ymax, xmin:xmax]
      crop_img = cv2.resize(crop_img, (250, 250))

    # 🎯 儲存帶有 YOLO 標註/畫框的照片
    save_filename = "latest_guava.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_filename)
    cv2.imwrite(save_path, annotated_img)

    image_public_url = f"https://guava-backend-ekg3.onrender.com/static/uploads/{save_filename}"
    print(
        f"\n📸 [ESP32-CAM Shot Received] Size: {w}x{h} px | YOLO Detected:"
        f" {detected}"
    )
    print(f"🔗 View marked image at: {image_public_url}\n")

    # E. Mask 計算
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    fruit_mask = gray > 15

    if np.sum(fruit_mask) > 0:
      score = float(np.std(gray[fruit_mask]))
    else:
      score = float(np.std(gray))

    # 釋放變數記憶體
    del img_resized, annotated_img, gray, crop_img, results
    gc.collect()

    # F. 動態分級
    t12 = model_config.get("threshold_1_to_2", 15.0)
    t23 = model_config.get("threshold_2_to_3", 9.0)

    if score >= t12:
      group = 1
      quality = "Excellent"
      desc = "果皮凹凸顆粒飽滿"
    elif score >= t23:
      group = 2
      quality = "Good"
      desc = "果皮質地中等"
    else:
      group = 3
      quality = "Smooth"
      desc = "果皮偏向光滑"

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


# 5. 健康檢查
@app.get("/")
def health_check():
  return {
      "status": "online",
      "message": "芭樂 AI 分級後端服務運作中 (加速防爆版20260922)",
      "current_config_version": model_config.get("version", "Unknown"),
  }
