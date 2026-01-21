# 4dt907 - Project in data intensive systems

## Roles

* Samuel - Scrum Master (DevOps)
* Nasser - Developer (full-stack)
* Emil - Developer (tester)
* Jesper - Developer (data scientist)

## Agile development

* Every other day stand-up via slack or in person on campus.
* Review of sprint during lecture weekly
* retrospective Mondays for approx 1 hour

## Refrence for repository setup

* Example [repository setup](https://github.com/SamuelFredricBerg/E2E-chat)
* Refrence for commit messages [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)
* Reference for pull request template ...
* Merging strategy ...
* Rulesets setup ...

## Planning 21/01/2026

* Fast-API
* CI/CD via github actions
* Docker
* React for frontend
* MLFlow
* Jupyter notebook for documentation
* Communication: Slack
* Request access to CScloud via lnu
* NodeJS version
* Python 3.12.X
* Docker version
* docker-compose version
* Secrets handling via github

## Branching strategy

* **main**: Production-ready code, protected branch
* **develop**: Integration branch for features
* **feature/**: Feature branches (e.g., `feature/user-authentication`)
* **bugfix/**: Bug fix branches (e.g., `bugfix/login-error`)
* **hotfix/**: Emergency fixes for production (e.g., `hotfix/security-patch`)

All branches should be merged via Pull Requests with code review.

## Local execution for development

### Prerequisites

* Docker and Docker Compose installed
* Python 3.12.x installed
* Node.js 20.x installed (for frontend development)
* Git configured

### Setup Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/sb224sc-HT22-VT27/4dt907.git
   cd 4dt907
   ```

2. For full stack development with Docker:

   ```bash
   cd src
   docker-compose up --build
   ```

3. For Python/ML development:

   ```bash
   cd src/backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python -m app.main
   ```

4. For frontend development:

   ```bash
   cd src/frontend
   npm install
   npm run dev
   ```

5. For Jupyter notebooks:

   ```bash
   pip install jupyter notebook
   jupyter notebook
   # Navigate to src/ml-research/
   ```

### Service URLs (when running with docker-compose)

* Frontend: <http://localhost:3000>
* Backend API: <http://localhost:8000>
* API Documentation: <http://localhost:8000/docs>
* MLflow UI: <http://localhost:5000>

## Project Structure (File Struct)

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
