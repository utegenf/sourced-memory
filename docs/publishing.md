# Publishing to PyPI

`sourced-memory` publishes via [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(GitHub Actions OIDC). No long-lived API token is stored in GitHub Secrets.
The workflow lives in `.github/workflows/release.yml` and runs when a tag
matching `v*` is pushed to the repository.

## First-time setup (one-time, ~5 minutes)

Everything below happens on pypi.org. Do this **before** pushing the first
tag; without it the workflow will fail at the publish step.

1. **Sign in** to https://pypi.org.
2. **Register a pending trusted publisher** at
   https://pypi.org/manage/account/publishing/ with:

   | Field                | Value               |
   |----------------------|---------------------|
   | PyPI Project Name    | `sourced-memory`    |
   | Owner (GitHub org)   | `utegenf`           |
   | Repository name      | `sourced-memory`    |
   | Workflow filename    | `release.yml`       |
   | Environment name     | `pypi`              |

3. **Create a GitHub Environment** named `pypi`
   (repo → Settings → Environments → New environment).
   Optionally require a manual approval before deployment; this is the
   place to gate releases on a maintainer review.

That's all. The token exchange happens automatically on every subsequent
release; the "pending publisher" is upgraded to a real publisher after the
first successful upload.

## Cutting a release

1. Bump the version in `pyproject.toml` and in `src/sourced_memory/__init__.py`
   (`__version__`). Alpha: `0.1.0a2` → `0.1.0a3`. Beta: `0.1.0b1`.
   Stable: `0.1.0`.
2. Commit with a message like `release: 0.1.0aN`.
3. Merge to `main` via the usual PR flow.
4. Tag `main`:

   ```bash
   git checkout main && git pull
   git tag v0.1.0aN
   git push origin v0.1.0aN
   ```

5. Watch the workflow at
   https://github.com/utegenf/sourced-memory/actions/workflows/release.yml.
6. Verify at https://pypi.org/project/sourced-memory/ that the new version
   is listed.
7. Announce (if applicable).

## What the workflow does

- Checks the tag version matches `pyproject.toml`'s `version` (mistaken
  tags fail loudly rather than publishing the wrong wheel).
- Installs the project + dev deps and runs `pytest`.
- Builds wheel and sdist with `python -m build`.
- Verifies the wheel contains only `sourced_memory/*` (no `research/`,
  no `tests/`).
- Publishes to PyPI via `pypa/gh-action-pypi-publish@release/v1` using
  trusted-publishing OIDC.

## Rollback

You cannot "delete" a version from PyPI once it has been uploaded, but
you can [yank](https://pypi.org/help/#yank) it, which prevents new
installs from picking it up while keeping it available to pinned users.

```
# yank on the PyPI website: project page → Manage → Releases → Options → Yank
```

Then bump the version and cut a new release. Do not reuse a version
number after yanking; PyPI rejects re-uploads.

## Notes

- **Never** commit a PyPI API token to the repository. Trusted publishing
  is used specifically to avoid this.
- The workflow only fires on `v*` tag pushes. Regular pushes to `main`
  will not accidentally trigger a release.
- If the tag was cut but the release failed, fix the underlying issue,
  delete the tag (`git tag -d vX; git push origin :vX`), and cut a new
  tag. PyPI is not consulted until the workflow reaches the publish
  step, so a failed early step is safe to retry.
