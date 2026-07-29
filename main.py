import os
import json
import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# 1. 全域變數與檔案路徑設定
MODEL_PATH = "best.pt"
CONFIG_PATH = "model_config.json"

yolo_model = None
model_config = {}

# 2. 使用現代 FastAPI 的 lifespan 管理伺服器生命週期（替代舊版 startup 事件）
@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, model_config
    
    # 載入 YOLO 模型
    if os.path.exists(MODEL_PATH):
        yolo_model = YOLO(MODEL_PATH)
        print("✅ YOLO 模型成功載入！")
    else:
        print("⚠️ 警告：找不到 best.pt，請確保模型檔放在當前目錄下")

    # 載入 K-Means 動態臨界值設定檔
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        print(f"✅ 設定檔載入成功 (版本: {model_config.get('version', 'Unknown')})")
    else:
        # 預設臨界值備案 (萬一找不到 json 檔時使用)
        model_config = {"threshold_1_to_2": 15.0, "threshold_2_to_3": 9.0}
        print("⚠️ 警告：找不到 model_config.json，已載入預設預備臨界值 (15.0, 9.0)")
    
    yield  # 伺服器啟動完成，開始接收請求
    
    # 伺服器關閉時釋放資源 (可留空)
    print("🛑 伺服器即將關閉...")

# 3. 初始化 FastAPI 應用
app = FastAPI(
    title="Guava Quality Assessment API",
    description="芭樂自動化分級與品質檢測後端服務",
    version="1.0.0",
    lifespan=lifespan
)

# 允許跨網域請求 (CORS)，方便手機 APP 或前端網頁連線
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 核心檢測 API 路由 (Endpoint)
@app.post("/api/v1/classify")
async def classify_guava(file: UploadFile = File(...)):
    """
    接收手機/前端上傳的照片檔案，進行即時辨識與品質分級
    """
    if yolo_model is None:
        raise HTTPException(status_code=500, detail="模型未就緒，請檢查伺服器端 best.pt 檔案")

    try:
        # A. 將上傳的圖片位元組轉換為 OpenCV 影像格式
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="無效的圖片檔案，無法解碼")

        # B. 影像等比例縮放 (維持與訓練/檢測集一致的 800px 寬度)
        target_w = 800
        scale = target_w / img.shape[1]
        target_h = int(img.shape[0] * scale)
        img_resized = cv2.resize(img, (target_w, target_h))
        h, w, _ = img_resized.shape

        # C. YOLO 偵測與正中央中心點定位
        results = yolo_model(img_resized, verbose=False)
        detected = False
        crop_img = None

        for r in results:
            for box in r.boxes:
                # 拆開雙重中括號
                x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
                
                # 計算正中央中心點 (暫不安裝 -20 位移)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 250x250 精準 ROI 區域裁切
                ymin, ymax = max(0, cy - 125), min(h, cy + 125)
                xmin, xmax = max(0, cx - 125), min(w, cx + 125)
                crop_img = img_resized[ymin:ymax, xmin:xmax]

                # 防呆防靠邊：長寬不足 250x250 則強制 Resize
                if crop_img.shape[0] != 250 or crop_img.shape[1] != 250:
                    crop_img = cv2.resize(crop_img, (250, 250))

                detected = True
                break
            if detected:
                break

        if not detected:
            # YOLO 辨識失敗時之中央裁切備用機制
            ymin, ymax = max(0, int(h/2)-125), min(h, int(h/2)+125)
            xmin, xmax = max(0, int(w/2)-125), min(w, int(w/2)+125)
            crop_img = img_resized[ymin:ymax, xmin:xmax]
            crop_img = cv2.resize(crop_img, (250, 250))

        # D. Mask 過濾黑邊背景與計算純果皮輝度標準差得分
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        fruit_mask = gray > 15
        
        if np.sum(fruit_mask) > 0:
            score = float(np.std(gray[fruit_mask]))
        else:
            score = float(np.std(gray))

        # E. 從動態設定檔 (model_config.json) 讀取臨界值進行分級
        t12 = model_config.get("threshold_1_to_2", 15.0)
        t23 = model_config.get("threshold_2_to_3", 9.0)

        if score >= t12:
            group = 1
            quality = "極佳"
            desc = "果皮凹凸顆粒飽滿"
        elif score >= t23:
            group = 2
            quality = "良好"
            desc = "果皮質地中等"
        else:
            group = 3
            quality = "平整"
            desc = "果皮偏向光滑"

        # F. 回傳結構化 JSON 給前端
        return {
            "status": "success",
            "data": {
                "filename": file.filename,
                "score": round(score, 2),
                "group": group,
                "quality": quality,
                "description": desc,
                "yolo_detected": detected,
                "version": model_config.get("version", "Unknown")
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"伺服器處理失敗: {str(e)}")

# 5. 健康檢查接口
@app.get("/")
def health_check():
    return {
        "status": "online", 
        "message": "芭樂 AI 分級後端服務運作中",
        "current_config_version": model_config.get("version", "Unknown")
    }