use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Resolve an optional dataset location relative to the workspace, not the crate's cwd.
pub fn data_root(workspace_root: &Path) -> PathBuf {
    resolve(
        workspace_root,
        std::env::var_os("OCRUS_DATA_DIR").as_deref(),
    )
}

fn resolve(workspace_root: &Path, configured: Option<&OsStr>) -> PathBuf {
    match configured.filter(|path| !path.is_empty()) {
        Some(path) => workspace_root.join(path),
        None => workspace_root.to_path_buf(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_or_empty_uses_workspace() {
        let root = Path::new("project");
        assert_eq!(resolve(root, None), root);
        assert_eq!(resolve(root, Some(OsStr::new(""))), root);
    }

    #[test]
    fn relative_location_uses_workspace() {
        let root = Path::new("project");
        assert_eq!(
            resolve(root, Some(OsStr::new("../datasets"))),
            root.join("../datasets")
        );
    }

    #[test]
    fn absolute_location_is_preserved() {
        let root = std::env::temp_dir().join("ocr-data");
        assert!(root.is_absolute());
        assert_eq!(resolve(Path::new("project"), Some(root.as_os_str())), root);
    }
}
