# Frontend Application

React + Vite frontend for the 4dt907 ML data-intensive system.
Includes a live squat analyzer powered by MediaPipe pose detection and a
prediction form that sends feature vectors to the backend.

## Getting Started

### Prerequisites

- [Node.js 22.22 (LTS)](https://nodejs.org/en/download)
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

### Supabase (optional)

The Squat Analyzer can persist recorded keypoint sessions to a Supabase database.
Set the following in your `.env` file (or in the hosting dashboard for production):

| Variable | Description |
| -------- | ----------- |
| `VITE_SUPABASE_URL` | Your Supabase project URL, e.g. `https://xyz.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Supabase anonymous/public API key |

If either variable is absent the app continues to work; the "Save to Database" button is
simply hidden. See `.env.example` for the required SQL to create the `squat_keypoints` table.

## Docker

Use the docker compose file to build entire project which uses the local Dockerfile for the frontend.

## Project Structure

```text
frontend/
├── src/
│   ├── App.jsx                 # Root component + tab navigation
│   ├── main.jsx                # Entry point
│   ├── apiBase.js              # Backend URL resolution (VITE_BACKEND_URL / proxy)
│   ├── supabaseClient.js       # Supabase client (null when env vars not set)
│   ├── featuresSchema.js       # Feature definitions for the prediction form
│   └── components/
│       ├── SquatAnalyzer.jsx   # MediaPipe live/video/image squat classifier
│       ├── Predict.jsx         # Expert-score prediction UI
│       ├── BackendStatus.jsx   # Backend health indicator
│       └── FeatureBuilder.jsx  # Dynamic feature input form
├── public/
├── index.html
├── vite.config.js
├── package.json
├── Dockerfile
└── README.md
```
