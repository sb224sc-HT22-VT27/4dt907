# Contributing

## Branching

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code (protected) |
| `develop` | Integration branch – target for most PRs |
| `feature/*` | New features (e.g. `feature/user-authentication`) |
| `bugfix/*` | Bug fixes (e.g. `bugfix/login-error`) |
| `hotfix/*` | Emergency production fixes |

All merges go through Pull Requests targeting `develop` (or `main` for hotfixes). Direct pushes to `main`/`develop` are not allowed.

## Commits

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard:

```
<type>(<scope>): <short summary>
```

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

Examples:
- `feat(backend): add weakest-link prediction endpoint`
- `fix(frontend): correct port in vite proxy config`
- `docs: update backend README`

Keep commits focused – each commit should contain only relevant changes for the issue it addresses.

## Pull Requests

Use the PR template when opening a pull request. At minimum:
- Link the related issue (`Closes #<issue-number>`)
- Fill in the **Type of Change** checklist
- Make sure all CI checks pass before requesting review

## CI Checks

The CI pipeline (`full-ci.yml`) runs on PRs targeting `main` and `develop`:

1. **backend-test** – `flake8` lint + `pytest` tests
2. **frontend-test** – `eslint` lint
3. **docker-build** – builds and smoke-tests both containers (runs after 1 & 2 pass)

Run these locally before pushing:

```bash
# Backend
cd src/backend
flake8 .
pytest tests/ -v

# Frontend
cd src/frontend
npm run lint
```

## Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`:

- **Bug Report** – for defects
- **Feature Request** – for new functionality
- **User Story** – for agile work items

## ML Experiments

When adding new ML notebooks:
- Work inside `src/ml-research/`
- Log all runs to MLflow/DagsHub
- Do **not** commit data files, model artifacts, or notebook outputs containing sensitive data
- Clear cell outputs before committing if they contain large blobs

