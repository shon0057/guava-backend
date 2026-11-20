from contextlib import asynccontextmanager
from datetime import datetime
import gc
import glob
import json
import os
import uuid
import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import torch
from ultralytics import YOLO

# 1. 解決 Render 系統目錄唯讀警告
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

MODEL_PATH = "best.pt"
CONFIG_PATH = "model_config.json"
UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

yolo_model = None
model_config = {}


# 輔助函式：延遲載入 YOLO 模型（避免 Render 啟動時 Port 綁定超時）
def get_yolo_model():
  global yolo_model
  if yolo_model is None and os.path.exists(MODEL_PATH):
    print("⏳ 正在載入 YOLO 多果種模型...")
    yolo_model = YOLO(MODEL_PATH)
    yolo_model.to("cpu")
    print(f"✅ YOLO 多果種模型成功載入！支援類別: {yolo_model.names}")
  return yolo_model


# 輔助函式：自動清理舊圖片，只保留最近 N 張
def cleanup_old_images(max_keep: int = 20):
  try:
    files = glob.glob(os.path.join(UPLOAD_DIR, "fruit_*.jpg"))
    if len(files) > max_keep:
      files.sort(key=os.path.getmtime)
      for old_file in files[:-max_keep]:
        os.remove(old_file)
  except Exception as e:
    print(f"⚠️ 清理舊圖時發生警告: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
  global model_config

  # 全域強制極簡模式：關閉梯度追蹤與單執行緒，降低模型記憶體腳印
  torch.set_grad_enabled(False)
  torch.set_num_threads(1)

  if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
      model_config = json.load(f)
    print(
        f"✅ 設定檔載入成功 (版本: {model_config.get('version', 'Unknown')})"
    )
  else:
    model_config = {
        "version": "2.0.0-default",
        "guava": {"threshold_1_to_2": 28.01, "threshold_2_to_3": 22.51},
        "watermelon": {"score_1_to_2": 13.5, "score_2_to_3": 9.5},
    }

  yield
  print("🛑 伺服器關閉")


app = FastAPI(
    title="Multi-Fruit AI Quality Assessment API",
    description="芭樂與小玉西瓜全自動品質分級後端服務",
    version="2.0.0-multi-fruit",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 演算法模組 A：芭樂質地分析 (中心 ROI + 灰階標準差)
# ==========================================
def analyze_guava(
    img_resized: np.ndarray,
    annotated_img: np.ndarray,
    box_coords: tuple,
    cfg: dict,
) -> dict:
  h, w = img_resized.shape[:2]
  x1, y1, x2, y2 = box_coords
  cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

  ymin, ymax = max(0, cy - 125), min(h, cy + 125)
  xmin, xmax = max(0, cx - 125), min(w, cx + 125)

  crop_img = img_resized[ymin:ymax, xmin:xmax].copy()
  cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

  if crop_img.shape[0] != 250 or crop_img.shape[1] != 250:
    crop_img = cv2.resize(crop_img, (250, 250))

  gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
  del crop_img
  fruit_mask = gray > 15

  if np.sum(fruit_mask) > 0:
    score = float(np.std(gray[fruit_mask]))
  else:
    score = float(np.std(gray))

  del gray, fruit_mask
  gc.collect()

  t12 = cfg.get("threshold_1_to_2", 28.01)
  t23 = cfg.get("threshold_2_to_3", 22.51)

  if score >= t12:
    group, quality, desc = (
        1,
        "Excellent",
        "果皮凹凸顆粒明顯飽滿 (特級脆口)",
    )
  elif score >= t23:
    group, quality, desc = 2, "Good", "果皮紋理適中 (優選品質)"
  else:
    group, quality, desc = 3, "Smooth", "果皮表面較平滑 (標準品質)"

  return {
      "fruit_type": "guava",
      "fruit_name_zh": "芭樂",
      "score": round(score, 2),
      "group": group,
      "quality": quality,
      "description": desc,
      "metrics": {"texture_std": round(score, 2)},
  }


# ==========================================
# 演算法模組 B：小玉西瓜黑紋分析 (內縮 ROI + Lab K-Means 粗細與占比)
# ==========================================
def analyze_watermelon(
    img_resized: np.ndarray,
    annotated_img: np.ndarray,
    box_coords: tuple,
    cfg: dict,
) -> dict:
  h, w = img_resized.shape[:2]
  x1, y1, x2, y2 = box_coords
  box_w, box_h = x2 - x1, y2 - y1

  # 1. 動態內縮 15% 作為 ROI，避開圓弧邊緣背景與反光
  pad_x, pad_y = int(box_w * 0.15), int(box_h * 0.15)
  xmin, ymin = max(0, x1 + pad_x), max(0, y1 + pad_y)
  xmax, ymax = min(w, x2 - pad_x), min(h, y2 - pad_y)

  # 防呆：若偵測框過小導致裁切無效，改用原偵測框
  if xmax <= xmin or ymax <= ymin:
    xmin, ymin, xmax, ymax = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

  # 畫出藍色分析區域框
  cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), (255, 180, 0), 2)

  roi_raw = img_resized[ymin:ymax, xmin:xmax].copy()
  roi_h, roi_w = roi_raw.shape[:2]

  # 統一縮放至 200x200 進行快速 K-Means，節省 Render 記憶體
  roi_small = cv2.resize(roi_raw, (200, 200), interpolation=cv2.INTER_AREA)
  del roi_raw

  # 2. 轉換至 Lab 色彩空間並執行 K-Means (K=2: 深色黑紋 vs 淺綠底色)
  lab_img = cv2.cvtColor(roi_small, cv2.COLOR_BGR2Lab)
  del roi_small
  # 取 L(亮度) 與 a(綠紅軸) 進行聚類，抗反光效果最佳
  pixel_values = lab_img[:, :, :2].reshape((-1, 2)).astype(np.float32)

  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  _, labels, centers = cv2.kmeans(
      pixel_values, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS
  )

  # 依據聚類中心的 L (亮度) 排序，自動鎖定亮度較低的那一群為「黑紋路」
  dark_idx = int(np.argmin(centers[:, 0]))
  light_idx = 1 - dark_idx
  stripe_mask = (labels.flatten() == dark_idx).reshape(200, 200).astype(
      np.uint8
  ) * 255

  # 形態學開運算消除細小噪點
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  stripe_mask = cv2.morphologyEx(stripe_mask, cv2.MORPH_OPEN, kernel)

  # 3. 計算黑紋特徵指標
  # (a) 黑紋面積占比 (%)
  stripe_ratio = float(np.sum(stripe_mask > 0) / stripe_mask.size * 100.0)

  # (b) 黑紋平均粗細 (利用距離變換計算半徑並乘以 2，再依比例還原回原圖像素尺度)
  dist_transform = cv2.distanceTransform(stripe_mask, cv2.DIST_L2, 5)
  scale_factor = ((roi_w / 200.0) + (roi_h / 200.0)) / 2.0
  if np.sum(dist_transform > 0) > 0:
    avg_width_px = float(
        np.mean(dist_transform[dist_transform > 0]) * 2.0 * scale_factor
    )
  else:
    avg_width_px = 0.0

  # (c) 黑紋與果皮底色的亮度對比度 (成熟度指標)
  contrast_diff = float(abs(centers[light_idx][0] - centers[dark_idx][0]))

  # 4. 視覺化加分：把 K-means 抓出的黑紋輪廓描繪在回傳圖片上 (黃色細線)
  mask_restored = cv2.resize(
      stripe_mask, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
  )
  contours, _ = cv2.findContours(
      mask_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  cv2.drawContours(
      annotated_img[ymin:ymax, xmin:xmax], contours, -1, (0, 255, 255), 1
  )

  del (
      lab_img,
      pixel_values,
      labels,
      stripe_mask,
      dist_transform,
      mask_restored,
  )
  gc.collect()

  # 5. 甜度等級判定
  t12 = cfg.get("score_1_to_2", 13.5)
  t23 = cfg.get("score_2_to_3", 9.5)

  if avg_width_px >= t12:
    group, quality, desc = (
        1,
        "Extra Sweet",
        f"黑紋粗壯清晰 (均寬 {avg_width_px:.1f}px)，預測特甜級 (≥12°Brix)",
    )
  elif avg_width_px >= t23:
    group, quality, desc = (
        2,
        "Sweet",
        f"黑紋分佈均勻 (均寬 {avg_width_px:.1f}px)，預測優選甜度 (10~11.9°Brix)",
    )
  else:
    group, quality, desc = (
        3,
        "Standard",
        f"黑紋較細窄 (均寬 {avg_width_px:.1f}px)，預測標準甜度 (<10°Brix)",
    )

  return {
      "fruit_type": "watermelon",
      "fruit_name_zh": "小玉西瓜",
      "score": round(avg_width_px, 2),
      "group": group,
      "quality": quality,
      "description": desc,
      "metrics": {
          "stripe_width_px": round(avg_width_px, 2),
          "stripe_ratio_pct": round(stripe_ratio, 2),
          "contrast_diff": round(contrast_diff, 2),
      },
  }


@app.post("/api/v1/classify")
async def classify_fruit(file: UploadFile = File(...)):
  model = get_yolo_model()
  if model is None:
    raise HTTPException(status_code=500, detail="模型未就緒，找不到 best.pt")

  try:
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    del contents, nparr
    gc.collect()

    if img_raw is None:
      raise HTTPException(status_code=400, detail="無效的圖片檔案")

    # 第一時間縮放至寬度 640px
    h_raw, w_raw = img_raw.shape[:2]
    target_w = 640
    target_h = int(h_raw * (640.0 / w_raw))

    img_resized = cv2.resize(
        img_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST
    )
    del img_raw
    gc.collect()

    h, w, _ = img_resized.shape

    # YOLO 推論
    with torch.inference_mode():
      results = model.predict(
          source=img_resized,
          imgsz=640,
          conf=0.45,
          max_det=1,
          verbose=False,
          save=False,
          show=False,
      )

    detected = False
    annotated_img = img_resized.copy()
    analysis_res = None

    if len(results) > 0 and len(results[0].boxes) > 0:
      box = results[0].boxes[0]
      x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
      conf = float(box.conf[0])
      cls_id = int(box.cls[0])
      fruit_type = model.names[cls_id].lower()
      detected = True

      # 依果種繪製外框 (芭樂綠框、西瓜橘紅框)
      box_color = (0, 255, 0) if "guava" in fruit_type else (0, 100, 255)
      cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 3)
      cv2.putText(
          annotated_img,
          f"{fruit_type.upper()} {conf:.2f}",
          (x1, max(25, y1 - 10)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.8,
          box_color,
          2,
      )

      # 自動分流至對應演算法
      if "watermelon" in fruit_type:
        analysis_res = analyze_watermelon(
            img_resized,
            annotated_img,
            (x1, y1, x2, y2),
            model_config.get("watermelon", {}),
        )
      else:
        analysis_res = analyze_guava(
            img_resized,
            annotated_img,
            (x1, y1, x2, y2),
            model_config.get("guava", {}),
        )

    del results
    gc.collect()

    # 若未偵測到任何水果的防呆機制
    if not detected:
      cv2.putText(
          annotated_img,
          "NO FRUIT DETECTED (Fallback Center)",
          (20, 40),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.75,
          (0, 0, 255),
          2,
      )
fallback_box = (
max(0, w // 2 - 125),
max(0, h // 2 - 125),
min(w, w // 2 + 125),
min(h, h // 2 + 125),
)
analysis_res = analyze_guava(
img_resized,
annotated_img,
fallback_box,
model_config.get("guava", {}),
)
analysis_res["fruit_type"] = "unknown"
analysis_res["fruit_name_zh"] = "未偵測到目標"
# 儲存標註結果圖
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
unique_id = uuid.uuid4().hex[:8]
save_filename = f"fruit_{timestamp}_{unique_id}.jpg"
save_path = os.path.join(UPLOAD_DIR, save_filename)
cv2.imwrite(save_path, annotated_img)
cleanup_old_images(max_keep=20)
del img_resized, annotated_img
gc.collect()
image_public_url = (
    f"https://onrender.com{save_filename}"
)
print(
f"📸 [Processed] Type: {analysis_res['fruit_type']} | Score:"
f" {analysis_res['score']} | Group: {analysis_res['group']}"
)
return {
"status": "success",
"data": {
"filename": file.filename,
"fruit_type": analysis_res["fruit_type"],
"fruit_name_zh": analysis_res["fruit_name_zh"],
"score": analysis_res["score"],
"group": analysis_res["group"],
"quality": analysis_res["quality"],
"description": analysis_res["description"],
"metrics": analysis_res["metrics"],
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
      "message": "芭樂 & 小玉西瓜雙果種 AI 分級後端服務運作中",
      "model_file_exists": os.path.exists(MODEL_PATH),
      "model_loaded": yolo_model is not None,
      "current_config_version": model_config.get("version", "Unknown"),
  }
