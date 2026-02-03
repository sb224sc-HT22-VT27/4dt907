# Frontend Application

React frontend application for 4dt907 ML data-intensive system.

## Features

- Web client for sending feature vectors to the backend and displaying predicted expert score
- Supports model variants via backend routes:
  - `POST /api/v1/predict/champion` (production/champion, typically `@prod`)
  - `POST /api/v1/predict/latest` (development/latest, typically `@dev`)
- Input validation guidance for the deployed regression model (expects **41** features)

## Getting Started

from src/frontend
npm ci
npm run dev

### Prerequisites

- Node.js 22.x (LTS)
- npm

### Installation

```bash
# Install dependencies
npm ci # or npm install
```

### Running the Application

```bash
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

The application will be available at <http://localhost:3000>

### Linting

```bash
# Check code quality
npm run lint
```

## Docker

```bash
# Build image
docker build -t 4dt907-frontend .

# Run container
docker run -p 3000:3000 4dt907-frontend
```

### API Integration

The frontend communicates with the backend API at <http://localhost:8000>

Key features:

### Components

## Project Structure (Update as needed)

```text
frontend/
├── src/
│   ├── App.jsx          # Main component
│   ├── App.css          # Component styles
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── index.html           # HTML template
├── vite.config.js       # Vite configuration
├── package.json
├── Dockerfile
└── README.md
```

## Configuration

### Vite Proxy

The Vite dev server is configured to proxy API requests:

## Contributing

Follow the [contribution guidelines](../../CONTRIBUTING.md) in the root repository.
