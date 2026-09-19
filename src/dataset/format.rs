use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DatasetFormat {
    Txt,
    Jsonl,
}

impl DatasetFormat {
    pub fn from_path(path: &Path) -> Result<Self, String> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("txt") => Ok(Self::Txt),
            Some("jsonl") => Ok(Self::Jsonl),
            _ => Err(format!(
                "cannot infer dataset format from {}",
                path.display()
            )),
        }
    }
}
