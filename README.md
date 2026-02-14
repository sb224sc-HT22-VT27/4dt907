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

## Local execution for development

### Prerequisites

* Docker and Docker Compose installed
* Python 3.12.x installed
* Node.js 22.x (LTS) installed (for frontend development)
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
   docker compose build
   docker compose up -d
   ```

3. [For backend development](src/backend/README.md)

4. [For frontend development](src/frontend/README.md)

5. [For ML notebooks](src/ml-research/README.md):

## Project Structure

```text
4dt907/
├── .github/                    # GitHub configuration
│   ├── workflows/              # CI/CD workflows
│   ├── ISSUE_TEMPLATE/         # Issue templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── CODEOWNERS
├── api/                        # Vercel serverless API
│   ├── index.py                # API entry point for Vercel
│   └── requirements.txt        # Python dependencies
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
├── .vercelignore               # Vercel deployment exclusions
├── vercel.json                 # Vercel deployment configuration
├── README.md
├── CONTRIBUTING.md
└── LICENSE
```

## Deployment

### Vercel Deployment

This project is configured for deployment on Vercel, supporting both frontend and backend:

#### Prerequisites

* Vercel account ([sign up](https://vercel.com/signup))
* Vercel CLI installed (optional): `npm install -g vercel`

#### Deployment Steps

1. **Import Project to Vercel:**
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "Add New" → "Project"
   - Import your GitHub repository

2. **Configure Environment Variables:**
   
   Set the following environment variables in your Vercel project settings:
   
   ```bash
   # Backend Configuration
   BACKEND_PORT=8080
   MLFLOW_TRACKING_URI=<your-mlflow-uri>
   MODEL_URI_PROD=<your-production-model-uri>
   MODEL_URI_DEV=<your-dev-model-uri>
   MODEL_URI_BACKUP=<your-backup-model-uri>
   WEAKLINK_MODEL_URI_PROD=<your-weaklink-prod-uri>
   WEAKLINK_MODEL_URI_DEV=<your-weaklink-dev-uri>
   WEAKLINK_MODEL_URI_BACKUP=<your-weaklink-backup-uri>
   
   # CORS Configuration (optional)
   ALLOWED_ORIGINS_PATTERN=^https://.*\.vercel\.app$
   PRODUCTION_URL=https://your-domain.com
   ```

3. **Deploy:**
   
   Vercel will automatically:
   - Build the React frontend from `src/frontend`
   - Deploy the FastAPI backend as serverless functions via `api/index.py`
   - Configure routing so `/api/*` requests go to the backend

4. **Verify Deployment:**
   
   Once deployed, test your endpoints:
   ```bash
   # Test frontend
   curl https://your-app.vercel.app
   
   # Test backend health
   curl https://your-app.vercel.app/api/health
   
   # Test API endpoints
   curl https://your-app.vercel.app/api/v1/predict/latest
   ```

#### Local Vercel Testing

To test the Vercel deployment locally:

```bash
# Install Vercel CLI
npm install -g vercel

# Run local Vercel dev server
vercel dev
```

This will start a local server that mimics the Vercel environment.

### Docker Deployment (Alternative)

For Docker-based deployment (development and testing):

```bash
cd src
docker compose build
docker compose up -d
```

Access:
- Frontend: http://localhost:3030
- Backend: http://localhost:8080
- API Docs: http://localhost:8080/docs
```
