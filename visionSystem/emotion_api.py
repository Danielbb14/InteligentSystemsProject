import os
import threading
import time
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION (4-Class RAF-DB) ---
MODEL_PATH = "vit_rafdb_4class.pth"
EMOTION_LABELS = ['angry', 'happy', 'neutral', 'sad']
WEBCAM_ID = 0
PROCESSING_INTERVAL = 0.5

# --- CAMERA SOURCE ---
# True  = pull frames from the Furhat robot's own camera (realtime API, port 9000)
# False = use the local webcam (cv2.VideoCapture(WEBCAM_ID))
USE_ROBOT_CAMERA = True
ROBOT_IP = "192.168.1.107"
ROBOT_REALTIME_KEY = os.getenv("FURHAT_REALTIME_KEY")

# --- Pydantic Schemas & Shared State ---
class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    timestamp: float
    message: str = "Success"

current_mood_state = {
    "emotion": "neutral",
    "confidence": 0.0,
    "timestamp": 0.0,
    "message": "Initializing..."
}
state_lock = threading.Lock()

# --- 1. MODEL CLASS (Core Logic) ---
class EmotionDetector:
    def __init__(self, model_path):
        # *** STABILITY FIX: FORCE CPU ***
        # This bypasses the persistent PyTorch/MPS loading error on your system.
        self.device = torch.device("cpu")
        print(f"Using device: {self.device} (Forced for loading stability)")

        num_classes = len(EMOTION_LABELS)
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        # *** The Loading Block ***
        try:
            # Load state dict onto CPU (most compatible method)
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            print(f"Model state loaded onto CPU successfully.")
        except Exception as e:
            # Re-raise with a clear message
            raise Exception(f"Failed to load model weights. Please check file integrity. Actual Error: {e}")

        # Move model to the selected device (CPU)
        self.model.to(self.device)
        self.model.eval()
        print("Model initialized and ready.")

        # RGB Transforms (matching your training script)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if self.face_cascade.empty():
            print("WARNING: Face cascade detector failed to load.")

    def preprocess_image(self, face_img):
        # Convert OpenCV's BGR format to RGB for PIL/PyTorch processing
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        return img_tensor

    def detect_emotion(self, face_img):
        img_tensor = self.preprocess_image(face_img)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = EMOTION_LABELS[predicted.item()]
        conf = confidence.item()
        return emotion, conf

    def process_frame(self, frame):
        """Performs face detection and emotion prediction on a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face Detection
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        if len(faces) == 0:
            return None, None, "No face detected."

        # Find the largest face and crop it
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_img = frame[y:y+h, x:x+w]

        # Run inference
        emotion, confidence = self.detect_emotion(face_img)
        return emotion, confidence, "Face detected and emotion predicted."

# --- 2. BACKGROUND THREAD FOR WEBCAM ---
def webcam_thread_function(detector_instance):
    """Continuously captures frames, runs the model, and updates the global state."""
    cap = cv2.VideoCapture(WEBCAM_ID)
    if not cap.isOpened():
        print("FATAL ERROR: Could not open webcam.")
        with state_lock: current_mood_state.update(message="FATAL: Camera not available", timestamp=time.time())
        return

    print("Webcam stream successfully started in background thread.")
    while True:
        ret, frame = cap.read()
        if ret:
            emotion, confidence, message = detector_instance.process_frame(frame)
            with state_lock:
                current_mood_state.update(
                    emotion=emotion if emotion else "neutral",
                    confidence=confidence if confidence else 0.0,
                    timestamp=time.time(),
                    message=message
                )
        else:
            with state_lock: current_mood_state.update(message="ERROR: Failed to read frame.", timestamp=time.time())

        time.sleep(PROCESSING_INTERVAL)

    cap.release()

# --- 2b. BACKGROUND THREAD FOR THE ROBOT'S CAMERA (realtime API) ---
def robot_camera_thread_function(detector_instance):
    """Pulls camera frames from the Furhat robot over the realtime API,
    decodes the base64 JPEG, runs the model, and updates the global state."""
    from furhat_realtime_api import FurhatClient
    try:
        client = FurhatClient(ROBOT_IP, ROBOT_REALTIME_KEY)
        client.connect()
        print(f"Connected to robot camera at {ROBOT_IP} (realtime API).")
    except Exception as e:
        print(f"FATAL ERROR: could not connect to robot realtime API: {e}")
        with state_lock:
            current_mood_state.update(emotion="error", message=f"Robot camera connect failed: {e}", timestamp=time.time())
        return

    while True:
        try:
            data = client.request_camera_once()
            b64 = data.get("image") if isinstance(data, dict) else None
            if not b64:
                with state_lock:
                    current_mood_state.update(message="No image in camera response.", timestamp=time.time())
            else:
                jpg = base64.b64decode(b64)
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
                if frame is None:
                    with state_lock:
                        current_mood_state.update(message="Frame decode failed.", timestamp=time.time())
                else:
                    emotion, confidence, message = detector_instance.process_frame(frame)
                    with state_lock:
                        current_mood_state.update(
                            emotion=emotion if emotion else "neutral",
                            confidence=confidence if confidence else 0.0,
                            timestamp=time.time(),
                            message=message
                        )
        except Exception as e:
            with state_lock:
                current_mood_state.update(message=f"Robot camera error: {e}", timestamp=time.time())
            time.sleep(1.0)
        time.sleep(PROCESSING_INTERVAL)

# --- 3. FASTAPI APP SETUP AND ENDPOINTS ---
detector: EmotionDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and start the webcam thread on startup."""
    global detector
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_full_path = os.path.join(script_dir, MODEL_PATH)
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found at: {model_full_path}")

        detector = EmotionDetector(model_full_path)

        cam_fn = robot_camera_thread_function if USE_ROBOT_CAMERA else webcam_thread_function
        thread = threading.Thread(target=cam_fn, args=(detector,))
        thread.daemon = True
        thread.start()
        print("Background webcam thread started.")
    except Exception as e:
        print(f"FATAL ERROR during startup: {e}")
        with state_lock:
            current_mood_state.update(
                emotion="error",
                confidence=0.0,
                timestamp=time.time(),
                message=f"Model Load Error: {e}"
            )

    yield
    # Shutdown logic would go here (after yield)


app = FastAPI(title="Real-time Emotion State API", lifespan=lifespan)


@app.get("/mood", response_model=EmotionResponse)
async def get_current_mood():
    """Returns the latest emotion state calculated by the background thread."""
    with state_lock:
        state = current_mood_state.copy()

    if state["emotion"] == "error":
        # Use 503 Service Unavailable if the model failed to load
        raise HTTPException(status_code=503, detail=state["message"])

    return EmotionResponse(**state)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

