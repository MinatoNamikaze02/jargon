#!/bin/bash

echo "🔍 Starting Privacy Policy NER Analyzer..."
echo "📁 Working directory: $(pwd)"

if [ ! -f "frontend/app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the frontend directory."
    exit 1
fi

if [ ! -d "../model" ] && [ ! -d "./model" ]; then
    echo "⚠️  Warning: No model directory found. Make sure you have trained a model first."
    echo "   Expected locations: ../model or ./model"
    exit 1
fi

# Start the server
echo "🚀 Starting FastAPI server..."
echo "🌐 Interface will be available at: http://localhost:8000"
echo "📊 API documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

poetry run python frontend/app.py
