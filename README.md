# 4dt907 - Project in data intensive systems

A full-stack ML application with a FastAPI backend and React frontend, deployed on Vercel.

## Quick Start

1. Clone the repository:

   ```bash
   git clone https://github.com/sb224sc-HT22-VT27/4dt907.git
   cd 4dt907
   ```

2. Copy `.env.example` to `.env` and fill in the values:

   ```bash
   cp .env.example .env
   ```

3. Choose development path:
   - [Backend development](src/backend/README.md)
   - [Frontend development](src/frontend/README.md)
   - [ML research notebooks](src/ml-research/README.md)
   - [Deployment strategies](DEPLOYMENT.md)

4. Or run the full stack with Docker:

   ```bash
   cd src
   docker compose build
   docker compose up -d
   ```

## Project Structure

```text
4dt907/
├── src/
│   ├── backend/           # FastAPI backend (Python)
│   ├── frontend/          # React frontend (Vite + Tailwind)
│   ├── ml-research/       # Jupyter notebooks for ML experiments
│   └── docker-compose.yml
├── api/                   # Vercel serverless entry point
├── .env.example           # Environment variable template
├── README.md
└── vercel.json            # Vercel configuration
```

## Branching Strategy

| Branch | Purpose |
| -------- | --------- |
| `main` | Production-ready code (protected) |
| `develop` | Integration branch |
| `feat/*` | New features |
| `bug/*` | Bug fixes |
| `fix/*` | Emergency production fixes |
| `<what>/*` | According to [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) |

All merges go through Pull Requests with code review.

## Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
