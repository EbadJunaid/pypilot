//! Pointing pyright and basedpyright at the project's `.venv`.
//!
//! Zed hands its Python language servers the interpreter of whichever toolchain
//! it selected, and in nested or multi-project worktrees that is sometimes not
//! the `.venv` PyPilot just set up (zed#49576, zed#59000, zed#50081), so every
//! installed import shows as unresolved. Pyright ranks `venvPath` + `venv` from
//! its config file above the editor's `python.pythonPath`, and basedpyright
//! inherits that, so a two-line config pins both to the right environment
//! whatever the toolchain picker says. ty reads neither key and is unaffected.
//!
//! Writing it is offered, never done silently: the file is usually committed,
//! and it overrides the toolchain selector for those servers. An existing
//! pyright configuration of any kind is left alone.

use std::path::{Path, PathBuf};

/// Where the pin would go.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinTarget {
    /// A `[tool.pyright]` table appended to the existing `pyproject.toml`.
    Pyproject,
    /// A new `pyrightconfig.json`, for projects without a `pyproject.toml`.
    PyrightConfig,
}

const PYPROJECT: &str = "pyproject.toml";
const PYRIGHTCONFIG: &str = "pyrightconfig.json";

/// Where the venv pin can be written, or `None` when it should not be offered:
/// no `.venv` to point at, or pyright is already configured (a
/// `pyrightconfig.json`, or a `[tool.pyright]`/`[tool.basedpyright]` table),
/// or `pyproject.toml` is not plain TOML we can safely extend.
pub fn pin_target(root: &Path) -> Option<PinTarget> {
    if !root.join(".venv").is_dir() || root.join(PYRIGHTCONFIG).exists() {
        return None;
    }
    let pyproject = root.join(PYPROJECT);
    if !pyproject.exists() {
        return Some(PinTarget::PyrightConfig);
    }
    let text = std::fs::read_to_string(&pyproject).ok()?;
    let table: toml::Table = text.parse().ok()?;
    if configures_pyright(&table) {
        return None;
    }
    Some(PinTarget::Pyproject)
}

fn configures_pyright(table: &toml::Table) -> bool {
    table
        .get("tool")
        .and_then(|t| t.as_table())
        .is_some_and(|tool| tool.contains_key("pyright") || tool.contains_key("basedpyright"))
}

/// Write the pin, re-checking first so nothing existing is ever overwritten.
/// Returns the file that was written.
pub fn write_pin(root: &Path) -> crate::Result<PathBuf> {
    match pin_target(root) {
        Some(PinTarget::Pyproject) => {
            let path = root.join(PYPROJECT);
            let text = std::fs::read_to_string(&path)?;
            let updated = with_pyright_table(&text)
                .ok_or_else(|| anyhow::anyhow!("pyproject.toml could not be extended safely"))?;
            std::fs::write(&path, updated)?;
            Ok(path)
        }
        Some(PinTarget::PyrightConfig) => {
            use std::io::Write;
            let path = root.join(PYRIGHTCONFIG);
            // `create_new` refuses to clobber a file that appeared meanwhile.
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?;
            file.write_all(b"{\n  \"venvPath\": \".\",\n  \"venv\": \".venv\"\n}\n")?;
            Ok(path)
        }
        None => anyhow::bail!("pyright is already configured for this project"),
    }
}

/// `text` with a `[tool.pyright]` table appended, in the file's own line
/// ending. `None` unless the result parses and carries exactly the pin, which
/// rules out layouts where appending a table would be invalid (an inline
/// `tool = { ... }`, say).
fn with_pyright_table(text: &str) -> Option<String> {
    let eol = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let mut out = text.to_string();
    if !out.is_empty() && !out.ends_with('\n') {
        out.push_str(eol);
    }
    if !out.is_empty() {
        out.push_str(eol);
    }
    for line in [
        "[tool.pyright]",
        "# Written by PyPilot so pyright/basedpyright use this project's .venv.",
        "venvPath = \".\"",
        "venv = \".venv\"",
    ] {
        out.push_str(line);
        out.push_str(eol);
    }

    let table: toml::Table = out.parse().ok()?;
    let pyright = table.get("tool")?.get("pyright")?;
    (pyright.get("venvPath")?.as_str() == Some(".")
        && pyright.get("venv")?.as_str() == Some(".venv"))
    .then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(name: &str, pyproject: Option<&str>) -> PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join(".venv")).unwrap();
        if let Some(text) = pyproject {
            std::fs::write(dir.join(PYPROJECT), text).unwrap();
        }
        dir
    }

    #[test]
    fn appends_the_pin_to_a_pyproject_without_pyright_config() {
        let original = "[project]\nname = \"demo\"\ndependencies = [\"requests\"]\n";
        let dir = project("pypilot-pyright-append", Some(original));
        assert_eq!(pin_target(&dir), Some(PinTarget::Pyproject));

        let path = write_pin(&dir).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with(original), "existing content is kept as is");
        let table: toml::Table = text.parse().unwrap();
        assert_eq!(table["tool"]["pyright"]["venv"].as_str(), Some(".venv"));
        assert_eq!(table["project"]["name"].as_str(), Some("demo"));

        // Configured now, so it is never offered again.
        assert_eq!(pin_target(&dir), None);
        assert!(write_pin(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn existing_pyright_config_is_never_touched() {
        for (name, text) in [
            (
                "pypilot-pyright-has-table",
                "[tool.pyright]\nstrict = [\"src\"]\n",
            ),
            (
                "pypilot-pyright-has-based",
                "[tool.basedpyright]\ntypeCheckingMode = \"basic\"\n",
            ),
            (
                "pypilot-pyright-dotted",
                "[tool]\npyright = { venv = \"env\" }\n",
            ),
        ] {
            let dir = project(name, Some(text));
            assert_eq!(pin_target(&dir), None, "{name}");
            assert!(write_pin(&dir).is_err());
            assert_eq!(std::fs::read_to_string(dir.join(PYPROJECT)).unwrap(), text);
            let _ = std::fs::remove_dir_all(&dir);
        }

        let dir = project(
            "pypilot-pyright-has-json",
            Some("[project]\nname = \"x\"\n"),
        );
        std::fs::write(dir.join(PYRIGHTCONFIG), "{}").unwrap();
        assert_eq!(pin_target(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nothing_is_offered_without_a_venv_or_with_unparsable_toml() {
        let dir = project("pypilot-pyright-no-venv", Some("[project]\n"));
        std::fs::remove_dir_all(dir.join(".venv")).unwrap();
        assert_eq!(pin_target(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);

        let dir = project("pypilot-pyright-broken", Some("[project\nname = "));
        assert_eq!(pin_target(&dir), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_project_without_pyproject_gets_a_pyrightconfig() {
        let dir = project("pypilot-pyright-json", None);
        assert_eq!(pin_target(&dir), Some(PinTarget::PyrightConfig));
        let path = write_pin(&dir).unwrap();
        assert_eq!(path, dir.join(PYRIGHTCONFIG));
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(json["venvPath"], ".");
        assert_eq!(json["venv"], ".venv");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_appended_table_follows_the_file_line_endings() {
        let crlf = with_pyright_table("[project]\r\nname = \"x\"").unwrap();
        assert!(crlf.ends_with("venv = \".venv\"\r\n"));
        assert!(!crlf.replace("\r\n", "").contains('\n'));
        // An inline `tool` table cannot take a `[tool.pyright]` header.
        assert_eq!(with_pyright_table("tool = { black = {} }\n"), None);
    }
}
