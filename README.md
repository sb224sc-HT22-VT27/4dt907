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

## Prerequisites

* Docker and Docker Compose installed for local deployment only
* Python 3.12.10 installed
* Node.js 22.22.0 (LTS) installed
* uv installed for vercel deployment (curl -LsSf <https://astral.sh/uv/install.sh> | sh)
* UNIX host or WSL for Windows users
* Vercel CLI (`vercel`) installed
* Git configured

### Setup Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/sb224sc-HT22-VT27/4dt907.git
   cd 4dt907
   ```

2. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env and fill in your values
   ```

3. For full stack development with Docker:

   ```bash
   cd src
   docker compose build
   docker compose up -d
   ```

4. [For backend development](src/backend/README.md)

5. [For frontend development](src/frontend/README.md)

6. [For ML notebooks](src/ml-research/README.md):

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
