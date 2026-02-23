# Frontend Application

React + Vite frontend for the 4dt907 ML data-intensive system.  
Sends feature vectors to the backend and displays predicted expert scores.

## Getting Started

### Prerequisites

- Node.js 22.x (LTS)
- npm

### Installation & Running

```bash
cd src/frontend
npm ci
npm run dev
```

The app will be available at <http://localhost:3030>

### Other Commands

```bash
npm run build    # Production build
npm run preview  # Preview production build
npm run lint     # Lint code
```

## Configuration

The Vite dev server proxies `/api/*` requests to the backend (default `http://localhost:8080`).  
Override by setting `BACKEND_URL` or `BACKEND_PORT` in your `.env` file at the project root.

## Docker

```bash
# From src/frontend
docker build -t 4dt907-frontend .
docker run -p 3030:3030 4dt907-frontend
```

Or use docker compose from `src/`:

```bash
docker compose up -d frontend
```

## Project Structure

```text
frontend/
├── src/
│   ├── App.jsx              # Root component
│   ├── main.jsx             # Entry point
│   ├── featuresSchema.js    # Feature definitions for the prediction form
│   └── components/
│       ├── Predict.jsx      # Main prediction UI
│       └── FeatureBuilder.jsx # Dynamic feature input form
├── public/
├── index.html
├── vite.config.js
├── package.json
├── Dockerfile
└── README.md
```

