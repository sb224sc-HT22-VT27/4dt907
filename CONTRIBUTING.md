# Contributing

## Branching Strategy

| Branch | Purpose |
| -------- | --------- |
| `main` | Production-ready code (protected) |
| `develop` | Integration branch |
| `feat/*` | New features |
| `bug/*` | Bug fixes |
| `fix/*` | Emergency fixes |
| `<what>/*` | According to [Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) |

All merges go through Pull Requests targeting `develop` (or `main` for hotfixes). Direct pushes to `main`/`develop` are not allowed.

## Commits

Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard:

```text
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

## Changelog

The `CHANGELOG.md` at the repository root tracks all notable changes to the project. Update it as part of every PR that introduces a meaningful change.

### When to add an entry

Add an entry whenever your PR includes one or more of:

- A new feature or API endpoint
- A bug fix that affects observable behaviour
- A refactoring that changes project structure or developer workflow
- A documentation update that changes how contributors work with the project

Minor style/typo fixes do not need a CHANGELOG entry.

### How to update it

1. Open `CHANGELOG.md`.
2. Under `## [Unreleased]`, add your change under the appropriate subsection:
   - **Added** – new functionality
   - **Changed** – modifications to existing functionality
   - **Fixed** – bug fixes
   - **Removed** – removed functionality
   - **Security** – security-related changes
3. Write a single, concise sentence per change. Start with an imperative verb (e.g. "Add …", "Fix …", "Change …").

```markdown
## [Unreleased]

### Added
- Add `/api/v1/ensemble` endpoint for ensemble model predictions

### Fixed
- Fix incorrect port reference in Vite proxy config
```

When a new version is released, the `[Unreleased]` block is renamed to the version number and date (e.g. `## [0.5.0] – 2026-03-01`), and a fresh `## [Unreleased]` section is opened at the top.

## ML Experiments

When adding new ML notebooks:

- Work inside `src/ml-research/`
- Log all runs to MLflow/DagsHub
- Do **not** commit data files, model artifacts, or notebook outputs containing sensitive data
- Clear cell outputs before committing if they contain large blobs
