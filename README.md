# Furhat Study Assistant with Emotion Detection

An intelligent study assistant system using the Furhat robot with real-time emotion detection and multimodal AI interaction powered by Google Gemini.

## System Overview

This project consists of three main components:
1. **Furhat Robot** - Physical/virtual robot interface
2. **Vision System** - Real-time emotion detection via webcam
3. **Main Interview Script** - Orchestrates the interaction using Gemini AI

## Prerequisites

- **Python 3.8+**
- **Furhat SDK** ([Download here](https://www.furhat.io/download/))
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))
- **Webcam** (for emotion detection)
- **Model File**: `vit_rafdb_4class.pth` (should be in `visionSystem/` directory)

## Installation

### 1. Install Furhat SDK

1. Download and install the Furhat SDK from [furhat.io](https://www.furhat.io/download/)
2. Launch Furhat SDK application
3. Start a virtual Furhat robot (or connect to a physical one)
4. Note the IP address (default: `localhost` for virtual robot)

### 2. Add the Furhat Remote API Skill

1. Open Furhat SDK
2. Go to the Skills section
3. Import the skill file: `skill/furhat-remote-api.skill`
4. Enable the Remote API skill on your Furhat robot

### 3. Set Up Python Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
API_KEY="your_google_gemini_api_key_here"
```

Replace `your_google_gemini_api_key_here` with your actual Google Gemini API key.

### 5. Verify Model File

Ensure the emotion detection model is present:
```bash
ls visionSystem/vit_rafdb_4class.pth
```

If missing, you'll need to obtain or train the ViT model for 4-class emotion detection (angry, happy, neutral, sad).

## Running the System

### Step 1: Start Furhat SDK
1. Launch the Furhat SDK application
2. Start your virtual or physical Furhat robot
3. Ensure the Remote API skill is active

### Step 2: Start the Vision System (Emotion Detection)

Open a terminal and run:
```bash
# Activate virtual environment if not already active
source venv/bin/activate

# Navigate to visionSystem directory
cd visionSystem

# Start the emotion detection API
python emotion_api.py
```

The vision system will:
- Start on `http://127.0.0.1:8000`
- Access your webcam
- Continuously process emotions in the background
- Expose a `/mood` endpoint

You can test it by visiting: `http://127.0.0.1:8000/mood`

### Step 3: Run the Main Interview Script

Open a **new terminal** (keep the vision system running):
```bash
# Activate virtual environment
source venv/bin/activate

# Run the main script
python mainInterview.py
```

## Configuration Options

### mainInterview.py Settings

Edit these variables in `mainInterview.py`:

```python
FURHAT_IP = "localhost"              # Change if using physical robot
QUESTIONS_TO_ASK = 5                 # Number of interaction rounds
USE_KEYBOARD = False                 # True for keyboard input, False for voice
MOOD_CLASSIFIER_ENABLED = True       # Enable/disable emotion detection
```

### Vision System Settings

Edit these in `visionSystem/emotion_api.py`:

```python
WEBCAM_ID = 0                        # Change if using different camera
PROCESSING_INTERVAL = 0.5            # Seconds between emotion checks
```

## Usage

Once everything is running:

1. The Furhat robot will greet you and ask for your name
2. You'll be asked what subject you're working on
3. The system will engage in 5 rounds of conversation
4. Your facial emotions are continuously detected and influence the robot's responses
5. At the end, you'll receive a summary

### Input Modes

- **Voice Mode** (default): Speak to the robot
- **Keyboard Mode**: Set `USE_KEYBOARD = True` and type responses

## Troubleshooting

### Furhat Connection Issues
- Verify Furhat SDK is running
- Check `FURHAT_IP` matches your robot's address
- Ensure Remote API skill is active

### Vision System Issues
- Check webcam is connected and not in use by other applications
- Verify model file exists: `visionSystem/vit_rafdb_4class.pth`
- Check API is running: `curl http://127.0.0.1:8000/mood`

### API Key Issues
- Verify `.env` file exists in project root
- Check API key is valid at [Google AI Studio](https://makersuite.google.com/)
- Ensure proper formatting: `API_KEY="your_key_here"`

### Dependencies Issues
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Project Structure

```
.
├── mainInterview.py          # Main orchestration script
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (API keys)
├── skill/
│   └── furhat-remote-api.skill  # Furhat Remote API skill (72MB)
├── visionSystem/
│   ├── emotion_api.py        # FastAPI emotion detection server
│   └── vit_rafdb_4class.pth  # Trained emotion detection model
└── output/                   # Output directory (if used)
```

## Dependencies

Key libraries:
- `furhat-remote-api` - Furhat robot control
- `google-generativeai` - Gemini AI integration
- `fastapi` + `uvicorn` - Emotion API server
- `opencv-python` - Webcam/face detection
- `torch` + `torchvision` - Deep learning inference
- `timm` - Vision Transformer model

See `requirements.txt` for complete list.

## Credits

Built for Uppsala University - Intelligent Robot Interaction course.
