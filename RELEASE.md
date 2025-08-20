# Release Process

This document describes the release process for StreamPoseML, including Docker image builds, PyPI publishing, and GitHub releases.

## Overview

StreamPoseML releases include:
- **Python package** published to PyPI
- **Docker images** pushed to DockerHub for all services (API, Web UI, MLflow)
- **GitHub release** with comprehensive release notes
- **Updated documentation** (CHANGELOG.md)

## Automated CI/CD Workflows

The project includes GitHub Actions workflows that automatically handle:

1. **CI Testing** (`ci.yml`) - Runs tests, linting, and type checking
2. **PyPI Publishing** (`publish-to-pypi.yml`) - Builds and publishes Python package
3. **Docker Images** - Built and pushed via manual `make build_images` command

## Standard Release Process

### Prerequisites

Before creating a release, ensure:

1. **All tests pass locally**:
   ```bash
   make test
   make lint
   ```

2. **Docker environment is set up**:
   ```bash
   # Ensure Docker buildx is configured for multi-platform builds
   docker buildx inspect --bootstrap
   ```

3. **GitHub CLI is authenticated**:
   ```bash
   gh auth status
   ```

### Step-by-Step Release

#### 1. Update Version

Update the version in `pyproject.toml`:
```toml
[project]
version = "0.2.3"  # Increment according to semantic versioning
```

#### 2. Update Documentation

Update `CHANGELOG.md` with the new release:
```markdown
## [0.2.3] - 2025-08-20

### Added
- New feature descriptions

### Fixed
- Bug fix descriptions

### Changed
- Changes to existing functionality
```

#### 3. Test Changes Locally

Run comprehensive tests to ensure everything works:
```bash
# Run all tests
make test

# Run linting and type checking
make lint

# Test Docker builds locally (optional but recommended)
make start-dev
# Verify application works correctly
make stop
```

#### 4. Commit Changes

Commit the version bump and changelog updates:
```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: Bump version to 0.2.3 and update changelog"
```

#### 5. Build and Push Docker Images

Build and push all Docker images to DockerHub:
```bash
make build_images
```

This command will:
- Build multi-platform images (linux/amd64, linux/arm64)
- Push to DockerHub with `:latest` tags:
  - `mrilikecoding/stream_pose_ml_api:latest`
  - `mrilikecoding/stream_pose_ml_web_ui:latest`
  - `mrilikecoding/stream_pose_ml_mlflow:latest`

#### 6. Create and Push Git Tag

Create a git tag and push everything:
```bash
git tag -a v0.2.3 -m "Release v0.2.3: Brief description of changes"
git push origin main
git push origin v0.2.3
```

#### 7. Create GitHub Release

Create the GitHub release with detailed notes:
```bash
gh release create v0.2.3 \
  --title "v0.2.3 - Release Title" \
  --notes "$(cat <<'EOF'
## 🚀 New Features
- Feature descriptions

## 🔧 Bug Fixes  
- Bug fix descriptions

## 🐳 Docker Images
Updated Docker images available on DockerHub:
- mrilikecoding/stream_pose_ml_api:latest
- mrilikecoding/stream_pose_ml_web_ui:latest
- mrilikecoding/stream_pose_ml_mlflow:latest

## 📦 Installation
\`\`\`bash
# Install Python package
pip install stream-pose-ml==0.2.3

# Or use Docker Compose
docker-compose -f docker-compose.build.yml up
\`\`\`
EOF
)"
```

#### 8. Monitor CI/CD Workflows

After pushing, monitor the automated workflows:
```bash
# Watch workflows
gh run watch

# Check workflow status
gh run list --limit 5
```

Ensure both CI and PyPI publishing workflows complete successfully.

#### 9. Verify Release

Verify the release was successful:

1. **Check PyPI**: Visit https://pypi.org/project/stream-pose-ml/
2. **Test installation**:
   ```bash
   pip install stream-pose-ml==0.2.3
   ```
3. **Verify Docker images**:
   ```bash
   docker pull mrilikecoding/stream_pose_ml_api:latest
   docker-compose -f docker-compose.build.yml up --no-build
   ```
4. **Test web application**: Navigate to http://localhost:3000

#### 10. Post-Release Cleanup

After workflows complete, there may be lock file updates:
```bash
# Check for any changes from CI workflows
git pull origin main

# If uv.lock was updated by workflows, commit it
git add uv.lock
git commit -m "chore: update uv.lock for version 0.2.3 release"
git push origin main
```

## Docker Image Versioning Strategy

### Current Approach
- All images use `:latest` tag
- Images are rebuilt and pushed for each release
- Multi-platform support (linux/amd64, linux/arm64)

### Future Improvements (Planned)
- Version-specific tags (e.g., `:v0.2.3`, `:0.2`, `:0`)
- Automated Docker builds via GitHub Actions
- Registry scanning and security updates

## Version Strategy

Follow [Semantic Versioning](https://semver.org/):

- **Major versions** (1.0.0): Breaking changes, API changes
- **Minor versions** (0.3.0): New features, backwards compatible
- **Patch versions** (0.2.3): Bug fixes, backwards compatible

### Examples
- Add new MLflow features → Minor version bump
- Fix Docker build issues → Patch version bump
- Change API endpoints → Major version bump

## Troubleshooting

### Common Issues

#### Docker Build Failures
```bash
# Check Docker buildx configuration
docker buildx ls

# Clean up build cache
docker system prune -a

# Rebuild images with verbose output
docker buildx build --progress=plain ...
```

#### CI/CD Workflow Failures
1. **PyPI Publishing Fails**:
   - Check if version already exists on PyPI
   - Verify `PYPI_API_KEY` secret is set
   - Review build logs: `gh run view [run-id] --log-failed`

2. **CI Tests Fail**:
   - Run tests locally: `make test && make lint`
   - Check for dependency conflicts
   - Verify Python version compatibility

3. **Docker Push Failures**:
   - Verify DockerHub credentials
   - Check network connectivity
   - Ensure multi-platform builder is available

#### Version Conflicts
```bash
# If you need to fix a release, create a patch version
git tag -d v0.2.3  # Delete local tag
git push origin :refs/tags/v0.2.3  # Delete remote tag
gh release delete v0.2.3  # Delete GitHub release

# Then fix issues and create new patch version v0.2.4
```

## Release Checklist

### Pre-Release
- [ ] All tests pass locally (`make test`)
- [ ] Code linting passes (`make lint`)
- [ ] Version updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] Local Docker builds work (`make start-dev`)

### Release
- [ ] Changes committed and pushed to main
- [ ] Docker images built and pushed (`make build_images`)
- [ ] Git tag created and pushed
- [ ] GitHub release created with detailed notes
- [ ] CI workflows monitored and successful
- [ ] PyPI publishing verified

### Post-Release
- [ ] PyPI package installation tested
- [ ] Docker images pulled and tested
- [ ] Web application functionality verified
- [ ] Lock files updated if needed
- [ ] Release announcement (if applicable)

### Verification
- [ ] Visit https://pypi.org/project/stream-pose-ml/ to confirm new version
- [ ] Test `pip install stream-pose-ml==[new-version]`
- [ ] Test `docker-compose -f docker-compose.build.yml up`
- [ ] Verify web application at http://localhost:3000
- [ ] Check GitHub release page for proper formatting

## Emergency Rollback

If a release has critical issues:

1. **Identify the issue severity**
2. **For critical bugs**:
   ```bash
   # Create hotfix patch release immediately
   git checkout v0.2.2  # Last known good version
   git checkout -b hotfix/0.2.4
   # Apply minimal fix
   # Follow release process for v0.2.4
   ```

3. **For non-critical issues**:
   - Document in GitHub issues
   - Plan fix for next regular release
   - Update documentation if needed

## Benefits of This Process

1. **Consistency**: Standardized steps reduce errors
2. **Quality**: Multiple verification stages catch issues
3. **Traceability**: Clear audit trail via git tags and GitHub releases
4. **Automation**: CI/CD handles testing and publishing
5. **Multi-Platform**: Docker images work across different architectures
6. **User Experience**: Users get reliable, tested releases

## Future Improvements

Planned enhancements to the release process:

- [ ] Automated Docker image builds via GitHub Actions
- [ ] Version-specific Docker tags
- [ ] Release candidate (RC) builds for testing
- [ ] Automated changelog generation from commit messages
- [ ] Integration tests in CI/CD pipeline
- [ ] Security scanning of Docker images
- [ ] Performance benchmarking across releases