# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in this project, please report it by:
1. Opening a private security advisory on GitHub
2. Contacting the team lead (Samuel) directly via Slack
3. Do NOT open a public issue for security vulnerabilities

## Security Updates

### MLflow Vulnerability Fixes (2026-01-21)

**Severity: CRITICAL**

Updated MLflow from 2.19.0 to 3.5.0 to address multiple security vulnerabilities:

1. **DNS Rebinding Attacks** (CVE)
   - Affected versions: < 3.5.0
   - Fixed in: 3.5.0
   - Impact: Lack of Origin header validation could allow DNS rebinding attacks

2. **Weak Password Requirements Authentication Bypass** (CVE)
   - Affected versions: < 2.22.0rc0
   - Fixed in: 2.22.0rc0 (and 3.5.0)
   - Impact: Weak password requirements could allow authentication bypass

3. **Model Creation Directory Traversal RCE** (CVE)
   - Affected versions: < 2.22.4
   - Fixed in: 2.22.4 (and 3.5.0)
   - Impact: Directory traversal vulnerability could lead to remote code execution

4. **Unsafe Deserialization** (CVE)
   - Affected versions: >= 0.5.0, <= 3.4.0
   - Fixed in: 3.5.0
   - Impact: Unsafe deserialization could allow code execution

**Actions Taken:**
- Updated `src/backend/requirements.txt`: mlflow 2.19.0 → 3.5.0
- Updated `src/ml-research/requirements.txt`: mlflow 2.19.0 → 3.5.0
- Updated `src/docker-compose.yml`: MLflow image v2.19.0 → v3.5.0
- Verified all tests pass with new version (6/6 tests passing)
- Verified linting passes (0 errors)

## Dependency Management

We use the following tools to manage and monitor dependencies:
- **pip**: Python package management
- **npm**: JavaScript package management
- **GitHub Dependabot**: Automated dependency updates and security alerts
- **GitHub Advisory Database**: Security vulnerability scanning

## Best Practices

1. **Secrets Management**
   - Never commit secrets, API keys, or passwords to Git
   - Use environment variables for sensitive configuration
   - Use GitHub Secrets for CI/CD pipelines

2. **Dependency Updates**
   - Regularly update dependencies to latest stable versions
   - Review security advisories for all dependencies
   - Test thoroughly after updates

3. **Code Review**
   - All code changes require PR review before merging
   - Security-sensitive changes require additional review
   - Use automated security scanning in CI/CD

4. **Access Control**
   - Follow principle of least privilege
   - Use CODEOWNERS for critical files
   - Enable branch protection on main branch

## Security Checklist for Contributors

- [ ] No secrets or sensitive data in code
- [ ] Dependencies are up-to-date and secure
- [ ] Input validation for user-provided data
- [ ] Proper error handling (no information leakage)
- [ ] Authentication and authorization where needed
- [ ] HTTPS/TLS for all external communications
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (proper output encoding)
- [ ] CSRF protection for state-changing operations

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Tools

- **flake8**: Python code quality and security checks
- **ESLint**: JavaScript code quality checks
- **GitHub Actions**: Automated security scanning in CI/CD
- **Docker**: Containerization for isolation

## Contact

For security concerns, contact:
- Team Lead: Samuel (via Slack)
- Repository: https://github.com/sb224sc-HT22-VT27/4dt907
