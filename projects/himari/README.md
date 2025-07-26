# Himari - AI-Powered YouTube Bot Detection Backend

Himari is the backend AI system for [noMoreSpam](https://github.com/misilelab/noMoreSpam), providing real-time bot detection for YouTube comments using advanced machine learning models.

## Overview

Himari is a cloud-native AI backend that analyzes YouTube comments to detect spam bots. It uses a custom-trained transformer model to evaluate both author names and comment content simultaneously, providing high-accuracy bot detection with real-time inference capabilities.

## Features

- **Advanced ML Model**: Custom-trained transformer model using KcELECTRA-base architecture
- **Dual-Input Analysis**: Evaluates both author names and comment content patterns
- **Real-time Inference**: Fast API responses with GPU acceleration
- **Smart Caching**: KV store with Blake3 hashing for performance optimization
- **Secure Authentication**: Argon2 password hashing and comprehensive API key validation
- **Scalable Infrastructure**: Built on Modal for serverless, auto-scaling deployment
- **High Accuracy**: Threshold-based classification with 90%+ confidence scoring

## Architecture

### Model Architecture
- **Base Model**: KcELECTRA-base transformer
- **Custom Layers**: Dual-input classification layers for author names and content
- **Output**: Binary classification (bot/human) with confidence scores
- **Threshold**: 0.9 confidence for bot classification

### System Components
- **FastAPI Web Server**: RESTful API endpoints
- **Modal Deployment**: Serverless GPU inference
- **KV Store**: Caching layer for performance optimization
- **Security Layer**: Argon2 hashing and API key validation

## API Endpoints

### POST `/evaluate`
Evaluates a batch of comments to determine if they are from bots.

**Request:**
```json
{
  "evaluate": [
    {
      "author_name": "string",
      "content": "string"
    }
  ],
  "api_key": "string"
}
```

**Response:**
```json
{
  "result": [0.95, 0.12, 0.87],
  "is_bot": [true, false, true]
}
```

### POST `/report`
Reports a comment as a bot for model improvement.

**Request:**
```json
{
  "author_name": "string",
  "content": "string",
  "is_bot": true,
  "api_key": "string"
}
```

**Response:**
```json
{
  "result": "success"
}
```

## Installation

### Prerequisites
- Python 3.13.5
- Modal account and CLI
- GPU access (recommended for production)

### Setup

1. **Clone the repository:**
```bash
git clone https://gith.misile.xyz/h3.git:/projects/himari.git
cd himari
```

2. **Install dependencies:**
```bash
uv sync
```

3. **Set up environment variables:**
Set up environment variables `API_KEY` from modal's secrets.

4. **Deploy to Modal:**
```bash
modal deploy main.py
```

### Model Training

The model uses a custom architecture with:
- **SpamUserClassificationLayer**: Dual-input neural network
- **SpamUserClassifier**: Main classification model
- **Training Data**: Custom dataset for YouTube bot detection

### Performance Optimization

- **Caching**: Blake3 hashed KV store for repeated queries
- **Batch Processing**: Efficient batch evaluation for multiple comments
- **GPU Acceleration**: CUDA support for fast inference
- **Memory Management**: Optimized model loading and inference

## Security

- **API Key Authentication**: Argon2 password hashing
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Built-in request limiting
- **Secure Headers**: CORS and security middleware
- **Minimal Permissions**: Chrome extension-only access

## Acknowledgments

- Inspired by characters from Blue Archive mobile game
- Built with Modal for scalable cloud deployment
- Uses KcELECTRA-base for Korean language optimization

