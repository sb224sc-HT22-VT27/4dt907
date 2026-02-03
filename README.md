# 4dt907 - Project in data intensive systems

## Refrence for repository setup

* Example [repository setup](https://github.com/SamuelFredricBerg/E2E-chat)
* Refrence for commit messages [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)

## Branching strategy

* **main**: Production-ready code, protected branch
* **develop**: Integration branch for features
* **feature/**: Feature branches (e.g., `feature/user-authentication`)
* **bugfix/**: Bug fix branches (e.g., `bugfix/login-error`)
* **hotfix/**: Emergency fixes for production (e.g., `hotfix/security-patch`)

All branches should be merged via Pull Requests with code review.

## Model Deployment & API Service

This project includes a complete ML model deployment solution with:
- FastAPI backend service that loads models from DagsHub MLflow
- React frontend for making predictions via web interface
- Docker Compose for easy deployment

📖 **[Read the Deployment Guide](DEPLOYMENT.md)** for complete setup and usage instructions.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/sb224sc-HT22-VT27/4dt907.git
   cd 4dt907
   ```

2. Configure environment variables:
   ```bash
   cd src
   cp .env.example .env
   # Edit .env with your DagsHub MLflow configuration
   ```

3. Start the services:
   ```bash
   docker compose up -d
   ```

4. Access the application:
   - Frontend: http://localhost:3030
   - Backend API: http://localhost:8080
   - API Docs: http://localhost:8080/docs

## Local Development

### Prerequisites

* Docker and Docker Compose installed
* Python 3.12.x installed
* Node.js 22.x (LTS) installed (for frontend development)
* Git configured
* DagsHub account with MLflow models

### Setup Steps

1. [Model Deployment & API Service](DEPLOYMENT.md) - Complete deployment guide

2. [For backend development](src/backend/README.md)

3. [For frontend development](src/frontend/README.md)

4. [For ML notebooks](src/ml-research/README.md):

## Project Structure

```text
4dt907/
├── .github/                    # GitHub configuration
│   ├── workflows/              # CI/CD workflows
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── src/                        # Source code
│   ├── backend/                # FastAPI backend service
│   │   ├── app/                # Application code
│   │   │   ├── api/            # API routes
│   │   │   ├── models/         # Data models
│   │   │   ├── services/       # Business logic
│   │   │   └── main.py         # Application entry point
│   │   ├── tests/              # Backend tests
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── frontend/               # React frontend application
│   │   ├── src/                # Frontend source code
│   │   ├── public/             # Static assets
│   │   ├── Dockerfile
│   │   └── package.json
│   ├── ml-research/            # Jupyter notebooks for assignments
│   │   ├── a1/                 # Assignment 1
│   │   ├── a2/                 # Assignment 2
│   │   └── ...
│   └── docker-compose.yml      # Multi-container orchestration
├── .gitignore
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```
