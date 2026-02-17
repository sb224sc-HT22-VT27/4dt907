# Vercel Deployment Workflow Setup

This guide explains how to set up automatic deployments to Vercel when changes are merged to the `main` branch.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com/signup)
2. **Vercel Project**: Import your GitHub repository to Vercel
3. **GitHub Repository Access**: Admin access to configure secrets

## Setup Steps

### 1. Get Vercel Credentials

You need three pieces of information from Vercel:

#### A. Vercel Token
1. Go to [Vercel Account Settings → Tokens](https://vercel.com/account/tokens)
2. Click "Create Token"
3. Give it a name (e.g., "GitHub Actions")
4. Set scope to your account/team
5. Copy the token (you won't see it again!)

#### B. Vercel Organization ID
```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Navigate to your project
cd /path/to/4dt907

# Link to Vercel project (if not already linked)
vercel link

# Get your Organization ID
cat .vercel/project.json
# Look for "orgId" field
```

#### C. Vercel Project ID
```bash
# From the same .vercel/project.json file
# Look for "projectId" field
```

Alternatively, you can find both IDs in your Vercel project settings:
- Go to your project on Vercel
- Settings → General
- Copy the Project ID and Organization ID

### 2. Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** and add each of these:

#### Required Secrets

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `VERCEL_TOKEN` | Your Vercel API token | `ABC123...` |
| `VERCEL_ORG_ID` | Your Vercel organization ID | `team_abc123...` |
| `VERCEL_PROJECT_ID` | Your Vercel project ID | `prj_abc123...` |

#### Optional Secrets (Environment Variables)

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `BACKEND_PORT` | Backend port (default: 8080) | `8080` |
| `MLFLOW_TRACKING_URI` | MLflow tracking server URI | `https://dagshub.com/user/repo.mlflow` |
| `MODEL_URI_PROD` | Production model URI | `models:/MyModel@prod` |
| `MODEL_URI_DEV` | Development model URI | `models:/MyModel@dev` |
| `MODEL_URI_BACKUP` | Backup model URI | `models:/MyModel@backup` |
| `WEAKLINK_MODEL_URI_PROD` | Weakest Link prod model | `models:/WeakLink@prod` |
| `WEAKLINK_MODEL_URI_DEV` | Weakest Link dev model | `models:/WeakLink@dev` |
| `WEAKLINK_MODEL_URI_BACKUP` | Weakest Link backup model | `models:/WeakLink@backup` |
| `PRODUCTION_URL` | Custom production domain | `https://your-domain.com` |

### 3. Configure GitHub Environment (Optional but Recommended)

For better control and approval gates:

1. Go to **Settings** → **Environments**
2. Click **"New environment"**
3. Name it `production`
4. Configure protection rules:
   - ✅ Required reviewers (optional)
   - ✅ Wait timer (optional)
   - ✅ Deployment branches: Only `main`

### 4. Test the Workflow

#### Option 1: Merge a PR to Main
1. Create a feature branch
2. Make some changes
3. Open a Pull Request to `main`
4. Once approved and merged, the workflow will trigger automatically

#### Option 2: Manual Trigger
1. Go to **Actions** tab in GitHub
2. Select **"Deploy to Vercel"** workflow
3. Click **"Run workflow"**
4. Select branch `main`
5. Click **"Run workflow"**

### 5. Monitor Deployment

1. Go to the **Actions** tab in GitHub
2. Click on the running workflow
3. Watch the deployment progress
4. Check the deployment summary for the URL

The workflow will output:
- ✅ Deployment URL
- ✅ Environment (production)
- ✅ Branch and commit SHA

## Workflow Behavior

### Automatic Triggers
- **Push to main**: Deploys automatically when code is merged to `main`
- **Manual trigger**: Can be triggered manually via GitHub Actions UI

### Deployment Steps
1. **Checkout code**: Gets the latest code from the repository
2. **Setup Node.js**: Installs Node.js 22
3. **Install Vercel CLI**: Installs the latest Vercel CLI
4. **Pull Vercel config**: Downloads Vercel project configuration
5. **Build artifacts**: Builds the project (frontend + backend)
6. **Deploy to Vercel**: Deploys the prebuilt artifacts
7. **Summary**: Outputs deployment information

### Environment Variables
All secrets configured in GitHub are passed to Vercel during the build step, making them available to your application.

## Troubleshooting

### Issue: "VERCEL_TOKEN not found"
**Solution**: Ensure you've added `VERCEL_TOKEN` to GitHub repository secrets.

### Issue: "Project not found"
**Solution**: 
1. Verify `VERCEL_ORG_ID` and `VERCEL_PROJECT_ID` are correct
2. Ensure the Vercel project exists and is linked to the repository

### Issue: "Build failed"
**Solution**:
1. Check the workflow logs in GitHub Actions
2. Ensure your `vercel.json` configuration is correct
3. Test the build locally: `vercel build --prod`

### Issue: "Environment variables not working"
**Solution**:
1. Verify secrets are added to GitHub
2. Check secret names match exactly (case-sensitive)
3. Ensure environment variables are passed in the workflow

## Alternative: Vercel Git Integration

Instead of using GitHub Actions, you can use Vercel's built-in Git integration:

### Pros of Vercel Git Integration:
- Automatic deployments on every push
- Preview deployments for every PR
- Simpler setup (no workflow file needed)
- Managed by Vercel

### Pros of GitHub Actions Workflow:
- More control over deployment process
- Can run additional checks before deployment
- Integration with existing CI/CD pipeline
- Can deploy to multiple environments

### Using Vercel Git Integration:
1. Import your repository to Vercel
2. Configure environment variables in Vercel dashboard
3. Vercel will automatically deploy on push to main

## Best Practices

1. **Use Environments**: Create separate environments for staging and production
2. **Protect Main Branch**: Require PR reviews before merging to main
3. **Test Locally First**: Always test `vercel build` locally before pushing
4. **Monitor Deployments**: Set up Vercel deployment notifications
5. **Use Preview Deployments**: Test changes in preview before merging
6. **Secret Management**: Rotate tokens periodically
7. **Deployment Approvals**: Use required reviewers for production

## Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel CLI Reference](https://vercel.com/docs/cli)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Vercel GitHub Integration](https://vercel.com/docs/concepts/git)

## Support

For issues with:
- **Workflow**: Check GitHub Actions logs
- **Vercel deployment**: Check Vercel deployment logs
- **Application errors**: Check Vercel function logs

---

**Last Updated**: 2026-02-17
**Workflow Version**: 1.0
