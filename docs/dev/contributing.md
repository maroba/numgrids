# Contributing

We welcome contributions to **numgrids** of all kinds -- bug reports, feature ideas,
documentation improvements, and code patches. This guide explains how to get involved.

## Reporting Bugs

If you encounter a bug, please check the existing
[GitHub Issues](https://github.com/maroba/numgrids/issues) first to make sure it has
not already been reported.

If the bug is new, [open an issue](https://github.com/maroba/numgrids/issues/new) and
include:

- A **clear title** and **description** of the problem.
- As much **relevant context** as possible (Python version, OS, numgrids version).
- A **minimal code sample** or **executable test case** that demonstrates the
  unexpected behavior.

## Feature Requests

Have an idea for a new feature or want to change existing behavior? Start a
conversation in the
[numgrids Discussion Forum](https://github.com/maroba/numgrids/discussions) before
writing code. This lets the community provide early feedback and avoids duplicate
effort.

```{note}
Please do **not** open a GitHub Issue for feature requests. Issues are reserved for
confirmed bug reports. Use [Discussions](https://github.com/maroba/numgrids/discussions)
instead.
```

## Pull Request Workflow

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:

   ```bash
   git clone https://github.com/<your-username>/numgrids.git
   cd numgrids
   ```

3. **Create a feature branch** from `main`:

   ```bash
   git checkout -b my-feature
   ```

4. **Develop** your changes. Follow the existing code style and patterns in the
   project -- consistency matters more than any specific formatter.

5. **Write tests** for any new functionality. Every new feature or bug fix should
   come with at least one test that verifies the expected behavior. See
   {doc}`testing` for details on how to run and write tests.

6. **Run the test suite** to make sure nothing is broken:

   ```bash
   python -m pytest tests
   ```

7. **Commit** your changes with a clear, descriptive commit message.

8. **Push** your branch to your fork and
   [open a Pull Request](https://github.com/maroba/numgrids/compare) against the
   `main` branch of the upstream repository. In the PR description, clearly explain
   the problem and your solution. Reference any related issue numbers.

## Code Style

- Follow the patterns and conventions already used in the codebase.
- Keep functions focused and well-documented with docstrings.
- Write tests for all new code paths.
- Use meaningful variable and function names.

## Questions?

If you have questions about the source code, usage, or anything else related to
numgrids, feel free to ask in the
[Discussion Forum](https://github.com/maroba/numgrids/discussions).

Thank you for helping improve numgrids!
