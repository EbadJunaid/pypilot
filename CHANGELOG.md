# Changelog

All notable changes to PyPilot's engine and Zed extension. Releases and their
full notes are on the [Releases page](https://github.com/pypilot-dev/pypilot/releases).

## [1.2.0]

- pixi support: pixi projects are detected and managed through pixi (`pixi install`,
  `pixi add --pypi`, `pixi add python=X.Y`); PyPilot never creates a `.venv`, runs uv
  or touches `pixi.lock` in them, and does nothing destructive when pixi is missing.
- The pixi environment (`.pixi/envs/default`) is used for import diagnostics, installed
  packages, GPU checks and the Python version; conda packages count as installed.
- PyPI dependencies of a pixi project are checked against its Python, and pixi's install
  errors are explained with the package to blame and the `pixi add python=…` fix.
- Doctor respects the pixi `python` spec when recommending a version and lists conda and
  PyPI dependencies separately.
- Fixed: version specifiers follow PEP 440, so `==2.15.0` no longer matches
  `2.15.0.post1` (pinned releases were analysed against the wrong files).
- Fixed: a release with no files for this platform (for example Linux-only wheels on
  Windows) is reported as uninstallable here instead of supporting every Python.

## [1.1.2]

- Moved to pypilot-dev organization; updated download URLs.
  - The Zed extension now downloads the helper from `pypilot-dev/pypilot`, and its
    download permission in `extension.toml` names that path.
  - Installed engines refresh the bundled data tables from the new location.
  - Repository links in the Cargo manifests and the data checker point at the new home.
- Added CONTRIBUTING, Code of Conduct and issue templates.
- No change in behavior.

Older versions redirect, so extensions and engines installed before this release keep
working. Zed extensions older than 1.1.2 can only download releases from the old path,
so they keep using the helper they already have until Zed updates them.
