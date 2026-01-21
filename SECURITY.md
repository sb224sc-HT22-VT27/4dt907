# Security Policy

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

## Security Tools

- **flake8**: Python code quality and security checks
- **ESLint**: JavaScript code quality checks
- **GitHub Actions**: Automated security scanning in CI/CD
- **Docker**: Containerization for isolation

## Contact

For security concerns, contact:

- Team Lead: Samuel (via Slack)
- Repository: <https://github.com/sb224sc-HT22-VT27/4dt907>
