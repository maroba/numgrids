# Releasing

Releases are published to [PyPI](https://pypi.org/project/numgrids/) automatically by the
GitHub Actions workflow `.github/workflows/release.yml` whenever a version tag is pushed.

## One-time setup

The workflow uses PyPI's [Trusted Publishing](https://docs.pypi.org/trusted-publishers/),
so no API token needs to be stored in the repository.

1. On pypi.org, open the project **numgrids** → *Settings* → *Publishing* and add a
   GitHub publisher:

   | Field       | Value         |
   |-------------|---------------|
   | Owner       | `maroba`      |
   | Repository  | `numgrids`    |
   | Workflow    | `release.yml` |
   | Environment | `pypi`        |

2. In the GitHub repository, create the environment **pypi** (*Settings* →
   *Environments*). Adding yourself as a required reviewer is recommended: every
   upload then waits for a manual approval.

3. Optional, for test uploads: register the same publisher on
   [test.pypi.org](https://test.pypi.org) with the environment **testpypi**, and
   create that environment on GitHub as well.

## Publishing a release

1. Update the version in `pyproject.toml` and add a section to the
   [changelog](changelog.md).
2. Merge the changes into `main`.
3. Create the tag, either on the command line:

   ```bash
   git tag v0.5.0
   git push origin v0.5.0
   ```

   or in the browser: on GitHub open *Releases* → *Draft a new release*, type the
   new tag `v0.5.0` under *Choose a tag* (select *Create new tag on publish*), keep
   the target branch `main`, enter a title such as `numgrids 0.5.0` and click
   *Publish release*.

The workflow then

* checks that the tag matches the version in `pyproject.toml`,
* runs the test suite,
* builds the sdist and the wheel, checks them with `twine check --strict` and
  smoke-tests the wheel in a clean environment,
* uploads the files to PyPI, and
* creates a GitHub release with the distribution files attached (or attaches the
  files to the release you created in the browser).

The release fails safely: if any check fails, nothing is uploaded.

## Test uploads

To try the pipeline without publishing a real release, run the workflow manually
(*Actions* → *Release* → *Run workflow*) with the target `testpypi`. Note that
PyPI and TestPyPI never accept the same version number twice.
