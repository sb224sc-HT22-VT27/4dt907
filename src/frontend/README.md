# Frontend Application

React + Vite frontend for the 4dt907 ML data-intensive system.
Sends feature vectors to the backend and displays predicted expert scores.

## Getting Started

### Prerequisites

- Node.js 22.22 (LTS)
- npm

### Installation & Running

```bash
cd src/frontend
npm install     # or npm ci
npm run dev
```

The application will be available at <http://localhost:3030>

### Other

```bash
npm run build   # Build for production
npm run preview # Preview production build
npm run lint    # Run linting
```

## Configuration

The Vite dev server proxies `/api/*` requests to the backend (default `http://localhost:8080`).
Override by setting `BACKEND_URL` or `BACKEND_PORT` in your `.env` file at the project root.

## Docker

Use the docker compose file to build entire project which uses the local Dockerfile for the frontend.

## Project Structure (Update as needed)

```text
frontend/
├── src/
│   ├── App.jsx                 # Root component
│   ├── main.jsx                # Entry point
│   ├── featuresSchema.js       # Feature definitions for the prediction form
│   └── components/
│       ├── Predict.jsx         # Main prediction UI
│       └── FeatureBuilder.jsx  # Dynamic feature input form
├── public/
├── index.html
├── vite.config.js
├── package.json
├── Dockerfile
└── README.md
```
