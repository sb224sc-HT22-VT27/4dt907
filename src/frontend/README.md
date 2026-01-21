# Frontend Application

React frontend application for 4dt907 ML data-intensive system.

## Features

## Getting Started

### Prerequisites

- Node.js 22.x (LTS)
- npm

### Installation

```bash
# Install dependencies
npm install
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

## Features

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
