# TruthLens AI 🔍

TruthLens AI is a real-time, multi-modal misinformation and deepfake detection platform. Built for a hackathon, it provides instant Explainable AI (XAI) analysis on text, images, videos, and URLs to combat the spread of fake news and manipulated media.

## Features ✨

- **Text Analysis**: Paste any news article, social media post, or WhatsApp forward. The NLP engine extracts linguistic features, cross-validates claims, and identifies suspicious phrases (supports Hinglish/Hindi!).
- **Image Deepfake Detection**: Upload an image to analyze it for high-frequency GAN fingerprints, noise pattern inconsistencies, and boundary blurring.
- **Video Deepfake Detection**: Drag and drop a video for frame-by-frame temporal consistency and deepfake artifact analysis.
- **URL Credibility Check**: Scrapes the domain and checks it against a known source database to instantly flag unverified domains and analyze the article's text.
- **Explainable AI (XAI)**: Every verdict (Real, Fake, or Suspicious) comes with human-readable reasoning and confidence scores.
- **Live Admin Dashboard**: A beautiful real-time dashboard for monitoring global analytics, trending misinformation keywords, and recent submissions.

## Tech Stack 🛠️

- **Backend**: FastAPI (Python), SQLite (Database)
- **Frontend**: Vanilla HTML5, CSS3 (Glassmorphism UI), JavaScript
- **AI / Processing**: OpenCV (Video/Image processing), Pillow (Image manipulation), NumPy/SciPy (FFT Analysis), NLP APIs (Gemini/Groq integration available)

## Quick Start 🚀

### 1. Backend Setup

Ensure you have Python 3.10+ installed.

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
*Note: Configure your `.env` file using `.env.example` as a template if you are setting up external API keys.*

### 2. Frontend Setup

In a new terminal window, serve the frontend files:

```bash
cd frontend

# Using python's built-in http server
python -m http.server 3000
```

Open `http://localhost:3000` in your web browser to access the Analyzer.  
Open `http://localhost:3000/dashboard.html` to access the Admin Dashboard.

## Project Structure 📁

```text
TruthLens-AI/
├── backend/
│   ├── api_routes/      # FastAPI endpoints for Text, Image, Video, URL, Dashboard
│   ├── database/        # SQLite setup and DB query layers
│   ├── models/          # Deepfake heuristics and AI model scripts
│   ├── utils/           # Helper scripts
│   ├── main.py          # FastAPI application entry point
│   └── requirements.txt # Python dependencies
└── frontend/
    ├── app.js           # Vanilla JS logic for uploading, API requests, and UI rendering
    ├── index.html       # The main multi-modal analyzer UI
    ├── dashboard.html   # Real-time analytics dashboard
    └── style.css        # Glassmorphism styling and responsive design
```

## Disclaimer
TruthLens AI is currently a prototype built for hackathon demonstration purposes. The deepfake heuristic models provide high-accuracy estimates but should not replace professional forensic analysis.
