# Changelog

All notable changes to PyPilot's engine and Zed extension. Releases and their
full notes are on the [Releases page](https://github.com/pypilot-dev/pypilot/releases).

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
