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

#### 4. Commit and Push Changes

Commit the version bump and changelog updates:
```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: Bump version to 0.2.3 and update changelog"
git push origin main
```

#### 5. Wait for CI to Pass

Before building Docker images, ensure all CI workflows complete successfully:
```bash
# Monitor CI workflows
gh run watch --exit-status

# Or check status manually
gh run list --limit 3
```

**Important**: Only proceed to Docker builds after confirming:
- ✅ All test jobs pass (Ubuntu, macOS, Python 3.10/3.11)
- ✅ Linting and type checking pass  
- ✅ Security checks pass

If CI fails, fix the issues, commit, push, and wait for CI to pass before continuing.

**Why wait for CI?**
- Prevents building and pushing Docker images with broken code
- Saves time and resources (multi-platform builds take 5-10 minutes)
- Ensures Docker images only contain tested, working code
- Follows best practice of never releasing untested code

#### 6. Build and Push Docker Images

Build and push all Docker images to DockerHub:
```bash
make build_images
```

This command will:
- Extract version from `pyproject.toml` automatically
- Build multi-platform images (linux/amd64, linux/arm64) using `--no-cache` for fresh builds
- Create and push both `:latest` and `:vX.X.X` version tags:
  - `mrilikecoding/stream_pose_ml_api:latest` and `mrilikecoding/stream_pose_ml_api:v[VERSION]`
  - `mrilikecoding/stream_pose_ml_web_ui:latest` and `mrilikecoding/stream_pose_ml_web_ui:v[VERSION]`
  - `mrilikecoding/stream_pose_ml_mlflow:latest` and `mrilikecoding/stream_pose_ml_mlflow:v[VERSION]`

**Important**: After the build completes, verify that both tag types are available:
```bash
# Check locally created tags
docker images | grep stream_pose_ml

# Verify DockerHub availability (may take a few minutes for propagation)
docker pull mrilikecoding/stream_pose_ml_api:v[VERSION]
docker pull mrilikecoding/stream_pose_ml_web_ui:v[VERSION]
docker pull mrilikecoding/stream_pose_ml_mlflow:v[VERSION]
```

If version tags are not immediately available on DockerHub, wait a few minutes for manifest propagation, or manually push if needed:
```bash
docker push mrilikecoding/stream_pose_ml_api:v[VERSION]
docker push mrilikecoding/stream_pose_ml_web_ui:v[VERSION]
docker push mrilikecoding/stream_pose_ml_mlflow:v[VERSION]
```

#### 7. Create and Push Git Tag

Create a git tag and push it to the remote repository:
```bash
git tag -a v0.2.3 -m "Release v0.2.3: Brief description of changes"
git push origin v0.2.3
```

#### 8. Create GitHub Release

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

#### 9. Monitor CI/CD Workflows

After pushing, monitor the automated workflows:
```bash
# Watch workflows
gh run watch

# Check workflow status
gh run list --limit 5
```

Ensure both CI and PyPI publishing workflows complete successfully.

#### 10. Verify Release

Verify the release was successful:

1. **Check PyPI**: Visit https://pypi.org/project/stream-pose-ml/
2. **Test installation**:
   ```bash
   pip install stream-pose-ml==[VERSION]
   ```
3. **Verify Docker images**:
   ```bash
   docker pull mrilikecoding/stream_pose_ml_api:latest
   docker-compose -f docker-compose.build.yml up --no-build
   ```
4. **Test web application**: Navigate to http://localhost:3000

#### 11. Post-Release Cleanup

**IMPORTANT**: After CI workflows complete, check for lock file updates from automated workflows:
```bash
# Check for any changes from CI workflows (especially uv.lock)
git pull origin main

# If uv.lock was updated by workflows, commit it
git add uv.lock
git commit -m "chore: update uv.lock for version [VERSION] release"
git push origin main
```

**Note**: The PyPI publishing workflow may update `uv.lock` with the newly published version. Always check for and commit these changes as part of the release process.

## Docker Image Versioning Strategy

### Current Approach
- Dual tagging: `:latest` and `:vX.X.X` version tags
- Version extracted automatically from `pyproject.toml`
- Images are rebuilt with `--no-cache` for each release to ensure fresh builds
- Multi-platform support (linux/amd64, linux/arm64)
- Manual verification step to ensure tag propagation

### Tag Strategy
- **`:latest`** - Always points to most recent release, used by `make start`
- **`:vX.X.X`** - Specific version tags for rollback scenarios (e.g., `:v0.2.3`)

### Future Improvements (Planned)
- Automated Docker builds via GitHub Actions
- Registry scanning and security updates
- Additional semantic version tags (e.g., `:0.3`, `:0`)

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
- [ ] **CI workflows pass completely** (all tests, linting, security)
- [ ] Docker images built and pushed (`make build_images`) **only after CI passes**
- [ ] Version tags verified on DockerHub (both `:latest` and `:vX.X.X`)
- [ ] Git tag created and pushed
- [ ] GitHub release created with detailed notes
- [ ] Additional CI workflows monitored (PyPI publishing)
- [ ] PyPI publishing verified

### Post-Release
- [ ] PyPI package installation tested
- [ ] Docker images pulled and tested
- [ ] Web application functionality verified
- [ ] **Lock files updated and committed** (`git pull` and check for `uv.lock` changes)
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
2. **For critical bugs requiring immediate rollback**:
   ```bash
   # Users can immediately switch to previous version
   docker pull mrilikecoding/stream_pose_ml_api:v[PREVIOUS_VERSION]
   docker pull mrilikecoding/stream_pose_ml_web_ui:v[PREVIOUS_VERSION]  
   docker pull mrilikecoding/stream_pose_ml_mlflow:v[PREVIOUS_VERSION]
   
   # Or create hotfix patch release
   git checkout v[PREVIOUS_VERSION]  # Last known good version
   git checkout -b hotfix/[PATCH_VERSION]
   # Apply minimal fix
   # Follow release process for v[PATCH_VERSION]
   ```

3. **For non-critical issues**:
   - Document in GitHub issues
   - Plan fix for next regular release
   - Update documentation if needed

**Note**: Version-specific Docker tags enable immediate rollback without rebuilding images.

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