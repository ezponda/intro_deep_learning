---
title: Pokemon Similarity Finder
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
---

# Pokemon Similarity Finder

This app helps you find the closest Pokemon match to any image you upload! Using advanced image similarity techniques, it compares your image against a database of Pokemon images to find the most similar match.

## Features
- Upload any Pokemon image
- Get instant similarity results
- Beautiful and intuitive interface
- Powered by deep learning

## How to Use
1. Upload an image of a Pokemon
2. Click "Find Similar Pokemon"
3. Get your results instantly!

## Technical Details
- Built with Streamlit
- Uses PyTorch for image processing
- Fast and efficient similarity matching

# Pokemon Similarity API

A FastAPI application that identifies Pokemon from images using similarity computation.

## Setup

### Option 1: Local Development

1. Create and activate a virtual environment using `uv`:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
uv pip install -r requirements.txt
```

3. Run the FastAPI application:
```bash
python main.py
```

4. Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

### Option 2: Docker

#### FastAPI Application

1. Build the Docker image:
```bash
docker build -t pokemon-similarity-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 pokemon-similarity-api
```

#### Streamlit Application

1. Build the Docker image:
```bash
docker build -t pokemon-similarity-streamlit -f Dockerfile.streamlit .
```

2. Run the container:
```bash
docker run -p 8501:8501 pokemon-similarity-streamlit
```

The FastAPI server will start on `http://localhost:8000` and the Streamlit app on `http://localhost:8501`.

## API Usage

The API provides two endpoints for Pokemon identification:

### 1. Upload Image File

```bash
curl -X POST -F "file=@path/to/your/image.jpg" http://localhost:8000/predict/upload
```

Response:
```json
{
    "pokemon": "pikachu"
}
```

### 2. Image URL

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/pokemon.jpg"}' \
  http://localhost:8000/predict/url
```

Response:
```json
{
    "pokemon": "pikachu"
}
```

## API Documentation

You can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
