#!/bin/bash

# Setup script for 4dt907 project
# This script helps set up the development environment

set -e

echo "🚀 Setting up 4dt907 project..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.12.x"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 20.x"
    exit 1
fi

echo "✓ Node.js version: $(node --version)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker is not installed. Docker is recommended for full stack development."
else
    echo "✓ Docker version: $(docker --version)"
fi

echo ""
echo "📦 Installing backend dependencies..."
cd src/backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
echo "✓ Backend dependencies installed"

echo ""
echo "📦 Installing frontend dependencies..."
cd ../frontend
npm install
echo "✓ Frontend dependencies installed"

cd ../..

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start with Docker: cd src && docker-compose up --build"
echo "   OR"
echo "2. Start backend: cd src/backend && source venv/bin/activate && python -m app.main"
echo "3. Start frontend: cd src/frontend && npm run dev"
echo ""
echo "Access points:"
echo "- Frontend: http://localhost:3000"
echo "- Backend API: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "- MLflow: http://localhost:5000"
