  #!/usr/bin/env python3
"""

445 
"""

import os
import io
import base64
import json
import time
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

# Web Framework
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request

# AI/ML Libraries
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
from PIL import Image
import qrcode
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import mediapipe as mp

# Firebase for real-time database (optional)
try:
    import firebase_admin
    from firebase_admin import credentials, db
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase not available - install firebase-admin if needed")

# Environment and Configuration
from dotenv import load_dotenv
import uvicorn
from pydantic import BaseModel

# Load environment variables
load_dotenv()



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Sortyx Medical Waste Classification API",
    description="Cloud-based medical waste classification and management system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories if they don't exist
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("uploads").mkdir(exist_ok=True)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Pydantic models for API requests/responses
class ClassificationRequest(BaseModel):
    image_base64: str
    bin_id: Optional[str] = None
    location: Optional[str] = "default"
    classification_method: Optional[str] = "model"  # "model" or "llm"

class ClassificationResponse(BaseModel):
    classification: str
    confidence: float
    item_name: str
    bin_color: str
    qr_code: Optional[str] = None
    explanation: str
    timestamp: str
    processing_time: float

class SensorData(BaseModel):
    sensor_id: str
    distance: float
    bin_level: float
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    location: str
    timestamp: str

class BinStatus(BaseModel):
    bin_id: str
    level: float
    status: str  # "normal", "warning", "full"
    last_updated: str

# Global variables for AI models
yolo_detection_model = None
yolo_classification_model = None
connected_websockets: List[WebSocket] = []

# Medical waste categories configuration (UPDATED FOR RECYCLABLE/NON-RECYCLABLE)
WASTE_CATEGORIES = {
    "Recyclable": {
        "color": "Green",
        "description": "Items that can be recycled: plastic bottles, metal cans, glass, paper, cardboard, electronics",
        "disposal_code": "REC"
    },
    "Non-Recyclable": {
        "color": "Black",
        "description": "Items that cannot be recycled: food waste, contaminated materials, styrofoam, ceramic",
        "disposal_code": "NR"
    }
}

class RecyclableWasteClassifier:
    """Enhanced recyclable waste classification system with Model + LLM support"""
    
    def __init__(self):
        self.load_models()
        self.configure_gemini()
        self.initialize_hand_detector()
        self.stats = {
            'total_classifications': 0,
            'category_counts': {category: 0 for category in WASTE_CATEGORIES.keys()},
            'daily_stats': {},
            'model_classifications': 0,
            'llm_classifications': 0
        }
    
    def initialize_hand_detector(self):
        """Initialize MediaPipe hand detection"""
        try:
            global mp_hands, mp_drawing
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe hand detection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing hand detection: {e}")
    
    def load_models(self):
        """Load YOLO models for detection and classification"""
        try:
            model_dir = Path("models")
            
            # Load detection model (for object detection)
            detection_model_path = model_dir / "yolov8n.pt"
            if detection_model_path.exists():
                global yolo_detection_model
                yolo_detection_model = YOLO(str(detection_model_path))
                logger.info("YOLO detection model loaded successfully")
            else:
                logger.warning(f"Detection model not found at {detection_model_path}")
            
            # Load RECYCLABLE classification model (YOUR best.pt model)
            # Try multiple paths to support both local and Docker environments
            classification_model_paths = [
                Path("/app/models/best.pt"),  # Docker path
                Path("models/best.pt"),  # Relative path
                Path("D:/Sortyx_Waste_App/best.pt"),  # Absolute Windows path
                Path("best.pt")  # Current directory
            ]
            
            classification_model_path = None
            for path in classification_model_paths:
                if path.exists():
                    classification_model_path = path
                    break
            
            if classification_model_path:
                global yolo_classification_model
                yolo_classification_model = YOLO(str(classification_model_path))
                logger.info(f"Recyclable classification model loaded successfully from {classification_model_path}")
            else:
                logger.warning(f"Classification model not found. Tried paths: {[str(p) for p in classification_model_paths]}")
                
        except Exception as e:
            logger.error(f"Error loading YOLO models: {e}")
    
    def configure_gemini(self):
        """Configure Google Gemini API"""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                logger.info("Gemini API configured successfully")
            else:
                logger.warning("GEMINI_API_KEY not found in environment variables")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {e}")
    
    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect objects in image using YOLO"""
        if yolo_detection_model is None:
            return {"error": "Detection model not loaded"}
        
        try:
            results = yolo_detection_model.predict(image, conf=0.25, iou=0.45)
            
            detections = []
            for r in results:
                if hasattr(r, 'boxes') and r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        class_id = int(box.cls[0])
                        class_name = r.names[class_id]
                        confidence = box.conf[0].item()
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "class_name": class_name,
                            "class_id": class_id,
                            "confidence": confidence,
                            "area": (x2 - x1) * (y2 - y1)
                        })
            
            return {"detections": detections, "count": len(detections)}
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return {"error": str(e)}
    
    def classify_with_yolo_model(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify waste using YOUR best.pt YOLO model"""
        if yolo_classification_model is None:
            logger.warning("YOLO classification model not loaded, falling back to LLM")
            return self.classify_with_gemini(image)
        
        try:
            # Run classification with YOUR model
            results = yolo_classification_model(image, verbose=False)
            
            if results and hasattr(results[0], 'probs') and results[0].probs is not None:
                top_class = results[0].probs.top1
                confidence = results[0].probs.top1conf.item()
                class_name = results[0].names[top_class]
                
                logger.info(f"YOLO Model Result: Class='{class_name}', Confidence={confidence:.2f}")
                
                # Map YOLO class to Recyclable/Non-Recyclable
                classification = self.map_yolo_class_to_category(class_name, confidence)
                
                return {
                    "classification": classification["category"],
                    "item_name": class_name.title(),
                    "explanation": f"Classified by AI model with {confidence*100:.1f}% confidence. {classification['reason']}",
                    "bin_color": WASTE_CATEGORIES[classification["category"]]["color"],
                    "disposal_code": WASTE_CATEGORIES[classification["category"]]["disposal_code"],
                    "confidence": confidence,
                    "method": "yolo_model"
                }
            else:
                logger.warning("No predictions from YOLO model, falling back to LLM")
                return self.classify_with_gemini(image)
                
        except Exception as e:
            logger.error(f"Error in YOLO classification: {e}")
            return self.classify_with_gemini(image)
    
    def map_yolo_class_to_category(self, class_name: str, confidence: float) -> Dict[str, Any]:
        """Map YOLO model output to Recyclable/Non-Recyclable"""
        class_lower = class_name.lower()
        
        # Define recyclable materials
        recyclable_keywords = [
            'plastic', 'bottle', 'can', 'metal', 'aluminum', 'glass', 
            'paper', 'cardboard', 'box', 'container', 'jar', 'tin',
            'e-waste', 'electronic', 'battery', 'phone', 'computer'
        ]
        
        # Define non-recyclable materials
        non_recyclable_keywords = [
            'food', 'organic', 'waste', 'trash', 'styrofoam', 'ceramic',
            'fabric', 'clothing', 'tissue', 'napkin', 'wrapper'
        ]
        
        # Check if item is recyclable
        is_recyclable = any(keyword in class_lower for keyword in recyclable_keywords)
        is_non_recyclable = any(keyword in class_lower for keyword in non_recyclable_keywords)
        
        if is_recyclable:
            return {
                "category": "Recyclable",
                "reason": "This item can be processed and reused."
            }
        elif is_non_recyclable:
            return {
                "category": "Non-Recyclable", 
                "reason": "This item cannot be recycled and should go to landfill."
            }
        else:
            # Default: if confidence is high, assume recyclable; otherwise non-recyclable
            if confidence > 0.7:
                return {
                    "category": "Recyclable",
                    "reason": "Likely recyclable based on AI analysis."
                }
            else:
                return {
                    "category": "Non-Recyclable",
                    "reason": "When uncertain, classify as non-recyclable for safety."
                }
    
    def classify_with_gemini(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify recyclable waste using Gemini AI (LLM method)"""
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Gemini prompt for RECYCLABLE waste classification
            prompt = """
            You are an expert waste recycling classifier. Analyze the image and classify the item as either RECYCLABLE or NON-RECYCLABLE.

            **RECYCLABLE ITEMS:**
            - Plastic bottles, containers, jugs (clean)
            - Metal cans (aluminum, steel, tin)
            - Glass bottles and jars
            - Paper (newspapers, magazines, office paper)
            - Cardboard boxes and packaging
            - E-waste (phones, batteries, electronics)

            **NON-RECYCLABLE ITEMS:**
            - Food waste and organic materials
            - Styrofoam and foam packaging
            - Ceramic items, pottery
            - Contaminated or greasy materials
            - Textiles and clothing
            - Mixed materials that can't be separated

            **INSTRUCTIONS:**
            - You MUST choose either "Recyclable" or "Non-Recyclable"
            - Be specific about what the item is
            - Explain why it belongs in that category

            **RESPONSE FORMAT:**
            Category: [Item Name]. [Brief explanation]

            **EXAMPLES:**
            - "Recyclable: Plastic Water Bottle. Clean plastic bottles can be melted down and reprocessed into new products."
            - "Non-Recyclable: Food waste Grease-stained cardboard contaminates recycling and should go to landfill."
            - "Recyclable: Aluminum Can. Metal cans are highly recyclable and valuable."
            - "Non-Recyclable: Styrofoam Container. Foam packaging cannot be recycled in most facilities."

            Analyze the image now and classify:
            """
            
            # Generate content using Gemini
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content([prompt, pil_image])
        
            if response and response.text:
                return self.parse_gemini_response(response.text, method="llm")
            else:
                return self.get_fallback_classification()
                
        except Exception as e:
            logger.error(f"Error in Gemini classification: {e}")
            return self.get_fallback_classification()
    
    def parse_gemini_response(self, text: str, method: str = "llm") -> Dict[str, Any]:
        """Parse Gemini response and extract classification details"""
        text_lower = text.lower()
        
        # Default classification
        classification = "Non-Recyclable"  # Default to safe option
        explanation = text
        item_name = "Unknown Item"
        
        # Enhanced classification detection for Recyclable/Non-Recyclable
        if any(word in text_lower for word in ["recyclable:", "recyclable", "can be recycled", "is recyclable"]):
            if "non-recyclable" not in text_lower and "not recyclable" not in text_lower:
                classification = "Recyclable"
        
        if any(word in text_lower for word in ["non-recyclable", "not recyclable", "cannot be recycled", "can't be recycled"]):
            classification = "Non-Recyclable"
        
        # Extract item name
        if ":" in text:
            try:
                parts = text.split(":", 1)
                if len(parts) > 1:
                    item_and_explanation = parts[1].strip()
                    first_sentence_end = item_and_explanation.find(".")
                    if first_sentence_end != -1:
                        item_name = item_and_explanation[:first_sentence_end].strip()
                    else:
                        item_name = item_and_explanation[:50].strip()
            except:
                pass
        
        # Get category details
        category_info = WASTE_CATEGORIES.get(classification, WASTE_CATEGORIES["Non-Recyclable"])
        
        return {
            "classification": classification,
            "item_name": item_name,
            "explanation": explanation,
            "bin_color": category_info["color"],
            "disposal_code": category_info["disposal_code"],
            "confidence": 0.85,  # High confidence for Gemini classifications
            "method": method
        }
    
    def get_fallback_classification(self) -> Dict[str, Any]:
        """Fallback classification when AI fails"""
        category_info = WASTE_CATEGORIES["Non-Recyclable"]
        return {
            "classification": "Non-Recyclable",
            "item_name": "Unknown Item",
            "explanation": "Could not classify item. Defaulting to non-recyclable for safety.",
            "bin_color": category_info["color"],
            "disposal_code": category_info["disposal_code"],
            "confidence": 0.50,
            "method": "fallback"
        }
    
    def generate_qr_code(self, classification_data: Dict[str, Any]) -> str:
        """Generate QR code for disposal tracking"""
        try:
            qr_data = {
                "id": str(uuid.uuid4()),
                "classification": classification_data["classification"],
                "item": classification_data["item_name"],
                "bin_color": classification_data["bin_color"],
                "disposal_code": classification_data["disposal_code"],
                "timestamp": datetime.now().isoformat(),
                "facility": "Sortyx Recycling Facility"
            }
            
            # Create QR code
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(json.dumps(qr_data))
            qr.make(fit=True)
            
            # Generate QR code image
            qr_image = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            qr_image.save(buffer, format="PNG")
            qr_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{qr_base64}"
            
        except Exception as e:
            logger.error(f"Error generating QR code: {e}")
            return None

# Initialize the classifier
classifier = RecyclableWasteClassifier()

# API Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "yolo_detection": yolo_detection_model is not None,
            "yolo_classification": yolo_classification_model is not None,
            "gemini_configured": bool(os.getenv('GEMINI_API_KEY'))
        }
    }

@app.post("/detect-hand")
async def detect_hand(request: ClassificationRequest):
    """Detect if a hand is present in the image using MediaPipe Hands (CPU only)"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Initialize MediaPipe Hands (CPU only, GPU disabled via environment variable)
        mp_hands = mp.solutions.hands
        
        hand_detected = False
        cropped_image = None
        bbox = None
        confidence = 0.0
        
        # Process with MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                hand_detected = True
                h, w, _ = image.shape
                
                # Get all hand landmarks to create bounding box
                all_x = []
                all_y = []
                
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        all_x.append(int(landmark.x * w))
                        all_y.append(int(landmark.y * h))
                
                # Create expanded bounding box to include object in hand
                margin = 80  # Margin to capture object being held
                x_min = max(0, min(all_x) - margin)
                x_max = min(w, max(all_x) + margin)
                y_min = max(0, min(all_y) - margin)
                y_max = min(h, max(all_y) + margin)
                
                # Crop the hand region with object
                cropped = image[y_min:y_max, x_min:x_max]
                
                # Convert cropped image to base64
                _, buffer = cv2.imencode('.jpg', cropped)
                cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                cropped_image = f"data:image/jpeg;base64,{cropped_base64}"
                
                bbox = {
                    "x_min": int(x_min),
                    "y_min": int(y_min),
                    "x_max": int(x_max),
                    "y_max": int(y_max)
                }
                
                # Get confidence from hand detection
                confidence = 0.95  # MediaPipe is very confident when it detects hands
                
                logger.info(f"Hand detected with {len(results.multi_hand_landmarks)} hand(s)")
        
        return {
            "hand_detected": hand_detected,
            "cropped_image": cropped_image,
            "bbox": bbox,
            "confidence": confidence,
            "num_hands": len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        }
                
    except Exception as e:
        logger.error(f"Hand detection error: {e}")
        return {
            "hand_detected": False,
            "cropped_image": None,
            "bbox": None,
            "confidence": 0.0,
            "num_hands": 0,
            "error": str(e)
        }

@app.post("/classify", response_model=ClassificationResponse)
async def classify_medical_waste(request: ClassificationRequest, background_tasks: BackgroundTasks):
    """Main classification endpoint - supports both YOLO model and LLM classification"""
    start_time = time.time()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64.split(',')[1] if ',' in request.image_base64 else request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.error("Failed to decode image")
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Choose classification method based on request parameter
        classification_method = (request.classification_method or "model").lower()
        
        logger.info(f"Starting classification with method: {classification_method}")
        
        classification_result = None
        
        try:
            if classification_method == "model":
                # Use YOLO model for classification
                classification_result = classifier.classify_with_yolo_model(image)
                classifier.stats['model_classifications'] += 1
                logger.info(f"Model classification: {classification_result.get('classification')} ({classification_result.get('confidence', 0):.2f})")
            else:
                # Use Gemini LLM for classification
                classification_result = classifier.classify_with_gemini(image)
                classifier.stats['llm_classifications'] += 1
                logger.info(f"LLM classification: {classification_result.get('classification')} ({classification_result.get('confidence', 0):.2f})")
        except Exception as class_error:
            logger.error(f"Classification error: {class_error}")
            # Use fallback classification
            classification_result = classifier.get_fallback_classification()
        
        if not classification_result or 'classification' not in classification_result:
            logger.error("Classification result is invalid")
            classification_result = classifier.get_fallback_classification()
        
        # Generate QR code
        qr_code = classifier.generate_qr_code(classification_result)
        
        # Update statistics
        classifier.stats['total_classifications'] += 1
        classifier.stats['category_counts'][classification_result['classification']] += 1
        
        processing_time = time.time() - start_time
        
        logger.info(f"Classification completed in {processing_time:.2f}s: {classification_result['classification']}")
        
        # Notify connected WebSocket clients
        background_tasks.add_task(notify_websocket_clients, {
            "type": "classification_complete",
            "data": classification_result
        })
        
        return ClassificationResponse(
            classification=classification_result["classification"],
            confidence=classification_result["confidence"],
            item_name=classification_result["item_name"],
            bin_color=classification_result["bin_color"],
            qr_code=qr_code,
            explanation=classification_result["explanation"],
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Classification endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/sensor/update")
async def update_sensor_data(sensor_data: SensorData):
    """Receive sensor data from ESP32"""
    try:
        # Process sensor data
        bin_status = process_sensor_data(sensor_data)
        
        # Notify WebSocket clients of sensor updates
        await notify_websocket_clients({
            "type": "sensor_update",
            "data": {
                "sensor_id": sensor_data.sensor_id,
                "bin_level": sensor_data.bin_level,
                "status": bin_status["status"]
            }
        })
        
        return {"status": "success", "bin_status": bin_status}
        
    except Exception as e:
        logger.error(f"Sensor update error: {e}")
        raise HTTPException(status_code=500, detail=f"Sensor update failed: {str(e)}")

@app.get("/bins/status")
async def get_bin_status():
    """Get current status of all bins"""
    try:
        # Mock bin status - replace with real database queries
        bins = [
            {"bin_id": "yellow_bin", "level": 45, "status": "normal", "last_updated": datetime.now().isoformat()},
            {"bin_id": "red_bin", "level": 78, "status": "warning", "last_updated": datetime.now().isoformat()},
            {"bin_id": "blue_bin", "level": 23, "status": "normal", "last_updated": datetime.now().isoformat()},
            {"bin_id": "black_bin", "level": 91, "status": "full", "last_updated": datetime.now().isoformat()}
        ]
        return {"bins": bins, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Get bin status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get system statistics"""
    return {
        "total_classifications": classifier.stats['total_classifications'],
        "category_breakdown": classifier.stats['category_counts'],
        "daily_stats": classifier.stats['daily_stats'],
        "timestamp": datetime.now().isoformat()
    }

# WebSocket for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connected_websockets.append(websocket)
    logger.info("WebSocket client connected")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
        logger.info("WebSocket client disconnected")

async def notify_websocket_clients(message: Dict[str, Any]):
    """Send message to all connected WebSocket clients"""
    if not connected_websockets:
        return
        
    for websocket in connected_websockets.copy():
        try:
            await websocket.send_json(message)
        except:
            connected_websockets.remove(websocket)

def process_sensor_data(sensor_data: SensorData) -> Dict[str, Any]:
    """Process sensor data and determine bin status"""
    level = sensor_data.bin_level
    
    if level >= 90:
        status = "full"
    elif level >= 75:
        status = "warning"  
    else:
        status = "normal"
    
    return {
        "bin_id": sensor_data.sensor_id,
        "level": level,
        "status": status,
        "last_updated": sensor_data.timestamp
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )