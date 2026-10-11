# Contributing to PyPilot

Thanks for your interest in PyPilot. It is a free, open-source project maintained
by [Abdullah Masood](https://github.com/Abdullah-Masood-05), and bug reports, ideas
and pull requests are all welcome.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## How this project is organized

PyPilot is split into a few repositories:

| Part | What it does |
|---|---|
| **This repo**, [`pypilot`](https://github.com/pypilot-dev/pypilot) | The Rust engine (the `pypilot` CLI and language server, in `helper/`) and the Zed extension (`extension/`). The environment logic, GPU/CUDA tables and diagnostics live here |
| [`pypilot-vscode`](https://github.com/pypilot-dev/pypilot-vscode) | The VS Code extension: status bar, commands, tasks, report panel, and downloading the engine |
| [`pypilot-pip`](https://github.com/pypilot-dev/pypilot-pip) | The `pypilot-cli` package on PyPI, a thin wrapper that downloads the engine |

Problems with *what PyPilot decides* (which Python version, which CUDA build, which
imports are flagged) belong here. Problems with *how it shows up in VS Code* belong in
`pypilot-vscode`, and problems with installing through pip belong in `pypilot-pip`. If
you are not sure, open the issue here and it will be moved.

## Licenses: two directories, two licenses

This repository contains two licenses, split by directory:

| Path | License | What it is |
|---|---|---|
| `helper/` | [AGPL-3.0-or-later](helper/LICENSE) | The `pypilot` engine: CLI and language server |
| `extension/` | [Apache-2.0](extension/LICENSE) | The Zed extension, a small WebAssembly shim |

The root [`LICENSE`](LICENSE) is the Apache-2.0 text. The `license` fields in
`helper/Cargo.toml` and `extension/Cargo.toml` say the same thing as the files above.

Why the split: Zed's extension registry only accepts permissive licenses for the code
it ships, and the shim is what Zed compiles and distributes, so it is Apache-2.0. The
engine does all the real work and runs as a separate process that the shim downloads,
which the registry rules exempt, so it is copyleft.

**Your contribution is licensed under the license of the directory you change.**
A change to `helper/` is AGPL-3.0-or-later, and a change to `extension/` is
Apache-2.0. Please do not copy code from `helper/` into `extension/`.

## Reporting a bug

Open an [issue](https://github.com/pypilot-dev/pypilot/issues/new?template=bug_report.md)
and include:

- What you did, what you expected, and what happened instead.
- Your OS, the editor you use (Zed, VS Code, or the CLI alone), and the engine version
  (`pypilot --version`).
- The output of `pypilot doctor` if the problem is about an environment.
- For CUDA or GPU problems, the GPU model and the NVIDIA driver version
  (from `nvidia-smi`).
- A minimal `requirements.txt` or `pyproject.toml` that reproduces it. Environment
  problems are much easier to fix when the dependencies involved are visible.

For security problems, please do not open a public issue. Contact the maintainer
through the GitHub profile linked above.

## Suggesting a feature

Open a [feature request](https://github.com/pypilot-dev/pypilot/issues/new?template=feature_request.md).
Describe the problem first, then the change you have in mind. PyPilot answers from
package metadata and local checks, with no AI service and no API keys, so ideas that
keep that property are the easiest to accept.

## Setting up for development

You need the stable Rust toolchain. [`rust-toolchain.toml`](rust-toolchain.toml) pins
`stable` with `rustfmt`, `clippy` and the `wasm32-wasip2` target, so `rustup` installs
them for you the first time you run `cargo` in the repo. Python 3.9 or later is only
needed for `scripts/check_data.py`.

```bash
git clone https://github.com/pypilot-dev/pypilot.git
cd pypilot
```

Before every commit that touches Rust, these must pass (CI runs the same checks):

```bash
cargo fmt --all -- --check
cargo clippy -p pypilot-helper --all-targets -- -D warnings
cargo test -p pypilot-helper
cargo build -p pypilot-zed --target wasm32-wasip2 --release
```

Tests are offline and use recorded PyPI responses as fixtures.

**Try the engine:** `cargo install --path helper --locked` installs `pypilot` on your
`PATH`. Both the Zed extension and the VS Code extension prefer a `pypilot` on `PATH`
over a downloaded release. After updating the engine, restart the language server
(in VS Code, run *Developer: Reload Window*).

**Try the Zed extension:** in Zed, run `zed: install dev extension` from the command
palette and select the `extension` directory.

## The data tables rule

The CUDA, driver and framework tables in `helper/data/*.json` are compiled into the
engine, and installed engines also re-download them from `main` about once a week.
That means a change merged to `main` reaches users without a release, and a broken
table breaks engines already in the field.

So, before any code change:

```bash
python scripts/check_data.py
```

Exit `0` means the tables are current. `1` means something is stale or the schema
changed, and the tables should be updated first, in their own `data: ...` commit.
`2` means a network error.

Keep the JSON schema backward compatible: only append rows or change values. Never
rename, remove or retype a field. [`AGENTS.md`](AGENTS.md) has the full rules, the
authoritative source for each table, and the update steps.

## Commits

- Make small, logical commits, each with a body that explains *why* the change is
  needed, not just what it does.
- Use conventional prefixes: `feat(scope):`, `fix:`, `test:`, `docs:`, `ci:`,
  `chore:`, `data:`.
- Use LF line endings.
- Signing commits is welcome but not required.

## Sending a pull request

1. Open an issue first for anything larger than a small fix, so we can agree on the
   approach before you spend time on it.
2. Fork the repo and create a branch from `main`.
3. If the change touches `helper/data/`, run `python scripts/check_data.py` and keep
   data changes in their own commit.
4. Keep the change focused. Match the style of the surrounding code, and explain *why*
   in comments rather than restating what the code does.
5. Run the four commands from the development section above.
6. Open the pull request, describe what changed and why, and link the issue. Mention
   anything you could not test (for example, a platform or GPU you do not have).

The maintainer reviews pull requests as time allows, so a short wait is normal.

## Releases

Releases are cut by the maintainer. Contributors do not need to bump versions or
create tags. A tagged engine release builds the binaries for each platform, and the
pip package and Zed registry entry are updated after that.
