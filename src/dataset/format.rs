use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DatasetFormat {
    Txt,
    Jsonl,
    Json,
    Csv,
    Aicorpus,
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
            Some("json") => Ok(Self::Json),
            Some("csv") => Ok(Self::Csv),
            Some("aicorpus") => Ok(Self::Aicorpus),
            _ => Err(format!(
                "cannot infer dataset format from {}",
                path.display()
            )),
        }
    }
}
