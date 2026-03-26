# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.7.0] – 2026-03-26

Assignment 7 – Unsupervised learning: PCA and clustering, plus split-deployment preparation.

### Added

- PCA and K-means/hierarchical clustering analysis for the dataset (`src/ml-research/a7/`)
- A7 report notebook documenting PCA, elbow method, and clustering results with visualisations
- `render.yaml` infrastructure-as-code for deploying the backend as a Render web service
- `BACKEND_URL` / `PRODUCTION_URL` environment variable support for split frontend–backend deployment

### Changed

- Frontend `apiBase.js` updated to support configurable backend URL for split deployment
- `DEPLOYMENT.md` expanded with Render and split-deployment instructions
- `.env.example` updated with additional environment variable specifications
- `vercel.json` archived (`vercel.old.json`) in favour of Render-based backend hosting

---

## [0.6.0] – 2026-03-11

Assignment 6 – SVM hyperparameter optimisation, model robustness improvements, and frontend enhancements.

### Added

- SVM classifier with `RandomizedSearchCV` + `GridSearchCV` hyperparameter optimisation (`src/ml-research/a6/`)
- 5-fold cross-validation comparison notebooks for regression and classification (`compare5kfolds.ipynb`, `compare5kfoldsA2.ipynb`)
- A6 report notebook and presentation slides documenting SVM results and model comparisons
- Frontend request/response duration timing displayed in the UI
- `imbalanced-learn` added to backend requirements for handling class imbalance

### Changed

- Bump `docker/setup-buildx-action` from v3 to v4 in CI/CD workflow

### Fixed

- Model info endpoints hardened: added error handling, logging, and HTTP 503 response when model info is unavailable
- Feature count detection improved for sklearn pipeline models (`n_features_in_` read from final estimator)
- Weakest-link model service updated with the same robust feature detection logic

---

## [0.5.0] – 2026-02-25

### Added

- Update READMEs for root, backend, frontend, and `ml-research` to match current project structure
- Clearer onboarding instructions across all modules
- This CHANGELOG file
- CONTRIBUTING guidelines for updating the CHANGELOG
- `DEPLOYMENT.md` with step-by-step deployment instructions linked from the root README
- A5 report notebook documenting ensemble/stacking evaluation for regression and classification
- Stacking ensemble implementation for A2 (regression) and A3 (classification) with supporting figures and metrics artifacts
- `.env` file specification added to the root README quick-start guide
- Updated and expanded documentation throughout the codebase.

### Changed

- Bump `actions/checkout` from v4 to v6 in CI/CD workflow
- Bump `actions/setup-python` from v5 to v6 in CI/CD workflow
- Bump `actions/setup-node` from v4 to v6 in CI/CD workflow
- Minor cleanup in `ml_utils.py`

---

## [0.4.0] – 2026-02-18

Assignment 4 – Maintenance sprint: quality assurance, Vercel deployment, and ML improvements.

### Added

- Vercel deployment configuration for frontend and backend (`vercel.json`, `api/` serverless entry-point)
- Cross-validation (CV) improvements to regression and classification models
- Dependabot configuration for automated dependency updates across Python, npm, and GitHub Actions ecosystems
- `pyproject.toml` / `uv` for Python dependency management

### Changed

- Project restructured under `src/` to support Vercel hosting (`src/backend/`, `src/frontend/`)
- Docker Compose updated to reflect new directory layout and correct environment variable loading
- Backend test suite refactored: one assertion per test, removed unused variables

### Fixed

- Docker health checks corrected (backend endpoint and BusyBox `wget` compatibility)
- Frontend proxy configuration aligned with updated backend port
- Caching issues in CI/CD workflows resolved
- Flake8 and Black formatting applied across all Python source files

---

## [0.3.0] – 2026-02-11

Assignment 3 – Classification model and weakest-link prediction endpoint.

### Added

- Classification model variants for weakest-link prediction (`src/ml-research/a3/`)
- `/api/v1/classify` endpoint returning the weakest-link classification result
- `/api/v1/model-info` endpoint exposing loaded model metadata
- A3 report notebook documenting classification iterations, API integration, and dependency management strategy
- Champion classification model selected and serialised for deployment

### Changed

- Backend extended with weakest-link classifier loaded from MLflow/DagsHub at startup
- CI/CD workflow improvements: Docker builds smoke-tested as part of the pipeline

---

## [0.2.0] – 2026-02-04

Assignment 2 – Regression model, ML pipeline, and initial API service.

### Added

- Linear regression model variants trained on the expert-score dataset (`src/ml-research/a2/`)
- MLflow experiment tracking integrated with DagsHub remote
- `/api/v1/predict` endpoint returning expert-score predictions from the champion regression model
- A2 report notebook documenting ML iterations, system architecture, and DevOps process
- React frontend extended to call the prediction endpoint and display results

### Changed

- FastAPI backend refactored to load the champion model from MLflow at startup
- Full client-server pipeline connected end-to-end

### Fixed

- Data contamination caused by `reset_index()` leaking the index column into training features
- MLflow tracking re-enabled after being accidentally disabled

---

## [0.1.0] – 2026-01-28

Assignment 1 – Project setup, full-stack hello-world, Docker, and CI/CD.

### Added

- FastAPI backend with versioned API endpoints (`/api/v1/hello`, `/api/v2/hello`)
- React frontend with version toggle and response rendering
- `Dockerfile` and `docker-compose.yml` for local and remote deployment
- GitHub Actions CI/CD pipeline (`full-ci.yml`): backend lint (`flake8`) + tests (`pytest`), frontend lint (`eslint`), and Docker build/smoke-test
- Initial project structure: `src/backend/`, `src/frontend/`, `src/ml-research/`, `src/scripts/`
- `CONTRIBUTING.md`, `README.md`, `SECURITY.md`, `.env.example`
