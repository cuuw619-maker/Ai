use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::Duration;
use url::Url;

const USER_AGENT: &str = "AiNet-DatasetDiscovery/1.0";
const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 256 * 1024 * 1024;
const ABSOLUTE_MAX_DOWNLOAD_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const MAX_PREVIEW_BYTES: usize = 128 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DatasetSourceKind {
    HuggingFace,
    GitHub,
    Wikimedia,
}
impl DatasetSourceKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::HuggingFace => "Hugging Face",
            Self::GitHub => "GitHub",
            Self::Wikimedia => "Wikimedia",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum DatasetState {
    Found,
    Queued,
    Downloading,
    AwaitingApproval,
    Downloaded,
    Preparing,
    Ready,
    UsedInTraining,
    Failed,
}
impl DatasetState {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Found => "FOUND",
            Self::Queued => "QUEUED",
            Self::Downloading => "DOWNLOADING",
            Self::AwaitingApproval => "APPROVAL REQUIRED",
            Self::Downloaded => "DOWNLOADED",
            Self::Preparing => "PREPARING",
            Self::Ready => "READY",
            Self::UsedInTraining => "USED IN TRAINING",
            Self::Failed => "FAILED",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetSearchFilters {
    pub query: String,
    pub language: String,
    pub topic: String,
    pub kind: String,
    pub size: String,
    pub min_quality: f32,
}
impl Default for DatasetSearchFilters {
    fn default() -> Self {
        Self {
            query: "general text".into(),
            language: "English".into(),
            topic: "General text".into(),
            kind: "Text".into(),
            size: "Small".into(),
            min_quality: 0.60,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetCandidate {
    pub id: String,
    pub name: String,
    pub source: DatasetSourceKind,
    pub source_url: String,
    pub download_url: Option<String>,
    pub preview_url: Option<String>,
    pub language: String,
    pub topic: String,
    pub kind: String,
    pub format: String,
    pub size_bytes: Option<u64>,
    pub size_label: String,
    pub description: String,
    pub license: String,
    pub author: String,
    pub quality_score: f32,
    pub small_model_recommended: bool,
    pub state: DatasetState,
    pub original_path: Option<PathBuf>,
    pub prepared_path: Option<PathBuf>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub enum DatasetEvent {
    SearchStarted,
    SourceSearching(DatasetSourceKind),
    Candidate(DatasetCandidate),
    SearchCompleted(usize),
    DownloadProgress {
        id: String,
        downloaded_bytes: u64,
        total_bytes: Option<u64>,
    },
    DownloadNeedsApproval {
        id: String,
        size_bytes: u64,
        limit_bytes: u64,
    },
    Preview {
        id: String,
        samples: Vec<String>,
    },
    Ready(DatasetCandidate),
    UsedForTraining(DatasetCandidate),
    Removed(String),
    Error {
        id: Option<String>,
        message: String,
    },
}

enum DatasetCommand {
    Search { filters: DatasetSearchFilters, auto: bool },
    Download { id: String, allow_large: bool },
    Prepare(String),
    Preview(String),
    UseForTraining(String),
    Pause,
    Stop,
    Remove(String),
}

pub trait DatasetSource: Send + Sync {
    fn kind(&self) -> DatasetSourceKind;
    fn search(&self, filters: &DatasetSearchFilters) -> Result<Vec<DatasetCandidate>, String>;
    fn list_datasets(&self) -> Result<Vec<DatasetCandidate>, String> {
        self.search(&DatasetSearchFilters::default())
    }
    fn get_metadata(&self, id: &str) -> Result<DatasetCandidate, String>;
}

pub struct HuggingFaceSource;
pub struct GitHubSource;
pub struct WikimediaSource;

impl DatasetSource for HuggingFaceSource {
    fn kind(&self) -> DatasetSourceKind {
        DatasetSourceKind::HuggingFace
    }

    fn search(&self, filters: &DatasetSearchFilters) -> Result<Vec<DatasetCandidate>, String> {
        let query = join_query(filters);
        let url = format!(
            "https://huggingface.co/api/datasets?search={}&limit=8",
            form_encode(&query)
        );
        let items = get_json(&url)?;
        let array = items.as_array().ok_or("Hugging Face returned invalid dataset list")?;
        let mut results = Vec::new();
        for item in array {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            let gated = item.get("gated").and_then(Value::as_bool).unwrap_or(false);
            if gated {
                continue;
            }
            let author = item.get("author").and_then(Value::as_str).unwrap_or_default();
            let downloads = item.get("downloads").and_then(Value::as_u64).unwrap_or(0);
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .or_else(|| item.get("cardData").and_then(|v| v.get("description")).and_then(Value::as_str))
                .unwrap_or("Public Hugging Face dataset.");
            let license = item
                .get("cardData")
                .and_then(|v| v.get("license"))
                .and_then(Value::as_str)
                .unwrap_or("Not specified")
                .to_string();
            let tags = item
                .get("tags")
                .and_then(Value::as_array)
                .map(|v| v.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            let quality = quality_for_metadata(description, downloads, &tags);
            if !language_matches(&filters.language, &tags, description) {
                continue;
            }
            let mut candidate = self.get_metadata(id)?;
            candidate.description = description.to_string();
            candidate.author = author.to_string();
            candidate.license = license;
            candidate.quality_score = quality;
            candidate.topic = infer_topic(&format!("{description} {tags}"));
            candidate.kind = if candidate.kind == "Unknown" { infer_kind(&candidate.description) } else { candidate.kind };
            if quality < filters.min_quality
                || !size_matches(candidate.size_bytes, &filters.size)
                || !kind_matches(&filters.kind, &candidate.kind)
                || !topic_matches(&filters.topic, &candidate.topic)
            {
                continue;
            }
            results.push(candidate);
        }
        Ok(results)
    }

    fn get_metadata(&self, id: &str) -> Result<DatasetCandidate, String> {
        let safe_id = id.trim().to_string();
        if safe_id.is_empty() || safe_id.contains(' ') {
            return Err("invalid Hugging Face dataset id".into());
        }
        let metadata_url = format!(
            "https://huggingface.co/api/datasets/{}/tree/main?recursive=true&expand=false&limit=2000",
            safe_id
        );
        let tree = get_json(&metadata_url)?;
        let mut selected: Option<(String, u64)> = None;
        if let Some(items) = tree.as_array() {
            for file in items {
                if file.get("type").and_then(Value::as_str) != Some("file") {
                    continue;
                }
                let Some(path) = file.get("path").and_then(Value::as_str) else {
                    continue;
                };
                let size = file.get("size").and_then(Value::as_u64).unwrap_or(0);
                if size == 0 || size > 64 * 1024 * 1024 {
                    continue;
                }
                if !allowed_data_extension(path) {
                    continue;
                }
                let rank = data_file_rank(path);
                if selected.as_ref().map(|(current, _)| data_file_rank(current) > rank).unwrap_or(true) {
                    selected = Some((path.to_string(), size));
                }
            }
        }
        let (path, size, format) = match selected {
            Some((path, size)) => {
                let format = display_format(&path);
                (Some(path), Some(size), format)
            }
            None => (None, None, "Unknown".into()),
        };
        let download_url = path.as_ref().map(|path| {
            format!(
                "https://huggingface.co/datasets/{}/resolve/main/{}",
                safe_id,
                path
            )
        });
        let kind = path
            .as_deref()
            .map(infer_kind_from_path)
            .unwrap_or("Unknown")
            .to_string();
        Ok(DatasetCandidate {
            id: format!("hf:{}", safe_id),
            name: safe_id.clone(),
            source: DatasetSourceKind::HuggingFace,
            source_url: format!("https://huggingface.co/datasets/{safe_id}"),
            download_url,
            preview_url: None,
            language: "Unknown".into(),
            topic: "General text".into(),
            kind,
            format,
            size_bytes: size,
            size_label: size.map(human_size).unwrap_or_else(|| "Unknown".into()),
            description: "Public Hugging Face dataset.".into(),
            license: "Not specified".into(),
            author: String::new(),
            quality_score: 0.60,
            small_model_recommended: size.map(|v| v <= 256 * 1024 * 1024).unwrap_or(false),
            state: DatasetState::Found,
            original_path: None,
            prepared_path: None,
            error: None,
        })
    }
}

impl DatasetSource for GitHubSource {
    fn kind(&self) -> DatasetSourceKind {
        DatasetSourceKind::GitHub
    }

    fn search(&self, filters: &DatasetSearchFilters) -> Result<Vec<DatasetCandidate>, String> {
        let query = format!("{} dataset", join_query(filters));
        let url = format!(
            "https://api.github.com/search/repositories?q={}&per_page=8",
            form_encode(&query)
        );
        let body = get_json(&url)?;
        let items = body
            .get("items")
            .and_then(Value::as_array)
            .ok_or("GitHub returned invalid repository search response")?;
        let mut results = Vec::new();
        for item in items {
            let Some(full_name) = item.get("full_name").and_then(Value::as_str) else {
                continue;
            };
            let description = item
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("Public dataset repository.");
            let quality = quality_for_metadata(
                description,
                item.get("stargazers_count").and_then(Value::as_u64).unwrap_or(0),
                "",
            );
            if let Ok(candidate) = self.get_metadata(full_name) {
                let candidate = DatasetCandidate {
                    description: description.to_string(),
                    quality_score: quality,
                    small_model_recommended: candidate.size_bytes.map(|v| v <= 128 * 1024 * 1024).unwrap_or(false),
                    ..candidate
                };
                if quality < filters.min_quality
                    || !size_matches(candidate.size_bytes, &filters.size)
                    || !kind_matches(&filters.kind, &candidate.kind)
                    || !topic_matches(&filters.topic, &candidate.topic)
                    || !language_matches(&filters.language, &candidate.language, &candidate.description)
                {
                    continue;
                }
                results.push(candidate);
            }
        }
        Ok(results)
    }

    fn get_metadata(&self, id: &str) -> Result<DatasetCandidate, String> {
        let repo = id.trim().trim_end_matches('/');
        let repo_meta = get_json(&format!("https://api.github.com/repos/{repo}"))?;
        let branch = repo_meta
            .get("default_branch")
            .and_then(Value::as_str)
            .unwrap_or("main");
        let description = repo_meta
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("Public GitHub dataset repository.");
        let license = repo_meta
            .get("license")
            .and_then(|v| v.get("spdx_id"))
            .and_then(Value::as_str)
            .unwrap_or("Not specified");
        let author = repo_meta
            .get("owner")
            .and_then(|v| v.get("login"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let tree = get_json(&format!(
            "https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        ))?;
        let mut selected: Option<(String, u64)> = None;
        if let Some(items) = tree.get("tree").and_then(Value::as_array) {
            for file in items {
                if file.get("type").and_then(Value::as_str) != Some("blob") {
                    continue;
                }
                let Some(path) = file.get("path").and_then(Value::as_str) else {
                    continue;
                };
                let size = file.get("size").and_then(Value::as_u64).unwrap_or(0);
                if size == 0 || size > 64 * 1024 * 1024 || !allowed_data_extension(path) {
                    continue;
                }
                let rank = data_file_rank(path);
                if selected.as_ref().map(|(current, _)| data_file_rank(current) > rank).unwrap_or(true) {
                    selected = Some((path.to_string(), size));
                }
            }
        }
        let (path, size, format) = match selected {
            Some((path, size)) => (Some(path.clone()), Some(size), display_format(&path)),
            None => (None, None, "Unknown".into()),
        };
        let download_url = path.as_ref().map(|path| format!("https://raw.githubusercontent.com/{repo}/{branch}/{path}"));
        let kind = path.as_deref().map(infer_kind_from_path).unwrap_or("Unknown").to_string();
        Ok(DatasetCandidate {
            id: format!("gh:{repo}"),
            name: repo.into(),
            source: DatasetSourceKind::GitHub,
            source_url: format!("https://github.com/{repo}"),
            download_url: download_url.clone(),
            preview_url: download_url,
            language: infer_language(description),
            topic: infer_topic(description),
            kind,
            format,
            size_bytes: size,
            size_label: size.map(human_size).unwrap_or_else(|| "Unknown".into()),
            description: description.into(),
            license: license.into(),
            author: author.into(),
            quality_score: 0.60,
            small_model_recommended: size.map(|v| v <= 128 * 1024 * 1024).unwrap_or(false),
            state: DatasetState::Found,
            original_path: None,
            prepared_path: None,
            error: None,
        })
    }
}

impl DatasetSource for WikimediaSource {
    fn kind(&self) -> DatasetSourceKind {
        DatasetSourceKind::Wikimedia
    }

    fn search(&self, filters: &DatasetSearchFilters) -> Result<Vec<DatasetCandidate>, String> {
        let lang = match filters.language.to_ascii_lowercase().as_str() {
            "russian" => "ru",
            "ukrainian" => "uk",
            _ => "en",
        };
        let index_url = format!("https://dumps.wikimedia.org/{lang}wiki/latest/");
        let body = get_text(&index_url, 2 * 1024 * 1024)?;
        let needle = format!("{lang}wiki-latest-pages-articles-multistream.xml.bz2");
        let has_dump = body.contains(&needle);
        let source_url = format!("https://dumps.wikimedia.org/{lang}wiki/latest/");
        let download_url = has_dump.then(|| format!("{source_url}{needle}"));
        let candidate = DatasetCandidate {
            id: format!("wm:{lang}wiki"),
            name: format!("Wikimedia {}wiki pages-articles", lang),
            source: DatasetSourceKind::Wikimedia,
            source_url,
            download_url,
            preview_url: None,
            language: filters.language.clone(),
            topic: "Knowledge".into(),
            kind: "Knowledge".into(),
            format: "XML+BZip2".into(),
            size_bytes: None,
            size_label: "Large (checked before download)".into(),
            description: "Latest public Wikimedia pages-articles multistream dump.".into(),
            license: "See Wikimedia project license".into(),
            author: "Wikimedia".into(),
            quality_score: 0.85,
            small_model_recommended: false,
            state: DatasetState::Found,
            original_path: None,
            prepared_path: None,
            error: None,
        };
        if !size_matches(candidate.size_bytes, &filters.size)
            || !kind_matches(&filters.kind, &candidate.kind)
            || !topic_matches(&filters.topic, &candidate.topic)
        {
            return Ok(Vec::new());
        }
        Ok(vec![candidate])
    }

    fn get_metadata(&self, id: &str) -> Result<DatasetCandidate, String> {
        self.search(&DatasetSearchFilters::default())?
            .into_iter()
            .find(|v| v.id == id)
            .ok_or_else(|| "Wikimedia dataset metadata not found".into())
    }
}

pub struct DatasetDiscovery {
    commands: Sender<DatasetCommand>,
    pub events: Receiver<DatasetEvent>,
    join: Option<thread::JoinHandle<()>>,
}
impl DatasetDiscovery {
    pub fn spawn(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("datasets/discovery"))
            .map_err(|e| format!("create dataset discovery storage: {e}"))?;
        let (commands, command_rx) = mpsc::channel();
        let (events, event_rx) = mpsc::channel();
        let join = thread::Builder::new()
            .name("ainet-dataset-discovery".into())
            .spawn(move || run_manager(root, command_rx, events))
            .map_err(|e| format!("spawn dataset discovery: {e}"))?;
        Ok(Self {
            commands,
            events: event_rx,
            join: Some(join),
        })
    }

    pub fn search(&self, filters: DatasetSearchFilters) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Search { filters, auto: false })
            .map_err(|e| format!("dataset discovery command: {e}"))
    }

    pub fn auto_discover(&self, filters: DatasetSearchFilters) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Search { filters, auto: true })
            .map_err(|e| format!("automatic dataset discovery command: {e}"))
    }

    pub fn pause(&self) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Pause)
            .map_err(|e| format!("dataset discovery pause command: {e}"))
    }

    pub fn stop(&self) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Stop)
            .map_err(|e| format!("dataset discovery stop command: {e}"))
    }
    pub fn download(&self, id: &str, allow_large: bool) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Download {
                id: id.into(),
                allow_large,
            })
            .map_err(|e| format!("dataset download command: {e}"))
    }
    pub fn prepare(&self, id: &str) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Prepare(id.into()))
            .map_err(|e| format!("dataset prepare command: {e}"))
    }
    pub fn preview(&self, id: &str) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Preview(id.into()))
            .map_err(|e| format!("dataset preview command: {e}"))
    }
    pub fn use_for_training(&self, id: &str) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::UseForTraining(id.into()))
            .map_err(|e| format!("dataset use-for-training command: {e}"))
    }
    pub fn remove(&self, id: &str) -> Result<(), String> {
        self.commands
            .send(DatasetCommand::Remove(id.into()))
            .map_err(|e| format!("dataset remove command: {e}"))
    }
}
impl Drop for DatasetDiscovery {
    fn drop(&mut self) {
        if let Some(join) = self.join.take() {
            let _ = self.commands.send(DatasetCommand::Remove("__shutdown__".into()));
            let _ = join.join();
        }
    }
}

fn run_manager(root: PathBuf, commands: Receiver<DatasetCommand>, events: Sender<DatasetEvent>) {
    let sources: Vec<Box<dyn DatasetSource>> = vec![
        Box::new(HuggingFaceSource),
        Box::new(GitHubSource),
        Box::new(WikimediaSource),
    ];
    let mut candidates = load_cached_candidates(&root);
    for candidate in candidates.values().cloned() {
        let _ = events.send(DatasetEvent::Candidate(candidate));
    }
    let mut download_queue = VecDeque::<(String, bool, bool)>::new();
    let mut paused = false;
    let mut stopped = false;

    loop {
        while let Ok(command) = commands.try_recv() {
            match command {
                DatasetCommand::Search { filters, auto } => {
                    paused = false;
                    stopped = false;
                    let mut total = 0usize;
                    let mut discovered_ids = Vec::new();
                    let _ = events.send(DatasetEvent::SearchStarted);
                    for source in &sources {
                        let _ = events.send(DatasetEvent::SourceSearching(source.kind()));
                        match source.search(&filters) {
                            Ok(items) => {
                                for mut item in items {
                                    item.state = DatasetState::Found;
                                    if let Some(existing) = candidates.get(&item.id) {
                                        if existing.original_path.is_some() {
                                            item.original_path = existing.original_path.clone();
                                            item.prepared_path = existing.prepared_path.clone();
                                            item.state = existing.state.clone();
                                        }
                                    }
                                    candidates.insert(item.id.clone(), item.clone());
                                    discovered_ids.push(item.id.clone());
                                    total += 1;
                                    let _ = events.send(DatasetEvent::Candidate(item));
                                }
                            }
                            Err(error) => {
                                let _ = events.send(DatasetEvent::Error {
                                    id: None,
                                    message: format!("{} search failed: {error}", source.kind().label()),
                                });
                            }
                        }
                    }
                    if auto {
                        let best_id = discovered_ids
                            .iter()
                            .filter_map(|id| candidates.get(id))
                            .filter(|candidate| {
                                candidate.download_url.is_some()
                                    && candidate.size_bytes.unwrap_or(u64::MAX)
                                        <= DEFAULT_MAX_DOWNLOAD_BYTES
                            })
                            .max_by(|a, b| {
                                a.quality_score
                                    .partial_cmp(&b.quality_score)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|candidate| candidate.id.clone());

                        if let Some(id) = best_id {
                            if let Some(candidate) = candidates.get_mut(&id) {
                                candidate.state = DatasetState::Queued;
                                candidate.error = None;
                                let _ = events.send(DatasetEvent::Candidate(candidate.clone()));
                                download_queue.push_back((id, false, true));
                            }
                        }
                    }
                    let _ = events.send(DatasetEvent::SearchCompleted(total));
                    persist_candidates(&root, candidates.values());
                }

                DatasetCommand::Download { id, allow_large } => {
                    if id == "__shutdown__" {
                        return;
                    }
                    paused = false;
                    stopped = false;
                    if let Some(candidate) = candidates.get_mut(&id) {
                        candidate.state = DatasetState::Queued;
                        candidate.error = None;
                        let _ = events.send(DatasetEvent::Candidate(candidate.clone()));
                        download_queue.push_back((id, allow_large, false));
                    } else {
                        let _ = events.send(DatasetEvent::Error { id: Some(id), message: "Dataset is not in discovery cache.".into() });
                    }
                }
                DatasetCommand::Prepare(id) => {
                    if let Some(candidate) = candidates.get(&id).cloned() {
                        match prepare_candidate(&root, &candidate, &events) {
                            Ok(updated) => {
                                candidates.insert(id.clone(), updated.clone());
                                let _ = events.send(DatasetEvent::Ready(updated));
                                persist_candidates(&root, candidates.values());
                            }
                            Err(error) => {
                                let mut failed = candidate.clone();
                                failed.state = DatasetState::Failed;
                                failed.error = Some(error.clone());
                                candidates.insert(id.clone(), failed.clone());
                                let _ = events.send(DatasetEvent::Candidate(failed));
                                let _ = events.send(DatasetEvent::Error { id: Some(id), message: error });
                            }
                        }
                    }
                }
                DatasetCommand::Preview(id) => {
                    if let Some(candidate) = candidates.get(&id).cloned() {
                        match preview_candidate(&candidate) {
                            Ok(samples) => {
                                let _ = events.send(DatasetEvent::Preview { id, samples });
                            }
                            Err(error) => {
                                let _ = events.send(DatasetEvent::Error { id: Some(id), message: error });
                            }
                        }
                    }
                }
                DatasetCommand::UseForTraining(id) => {
                    if let Some(candidate) = candidates.get_mut(&id) {
                        if candidate.prepared_path.is_none() {
                            let _ = events.send(DatasetEvent::Error { id: Some(id), message: "Prepare the dataset before using it for training.".into() });
                        } else {
                            candidate.state = DatasetState::UsedInTraining;
                            let updated = candidate.clone();
                            candidates.insert(updated.id.clone(), updated.clone());
                            let _ = events.send(DatasetEvent::UsedForTraining(updated));
                            persist_candidates(&root, candidates.values());
                        }
                    }
                }
                DatasetCommand::Pause => {
                    paused = true;
                }
                DatasetCommand::Stop => {
                    paused = true;
                    stopped = true;
                    download_queue.clear();
                }
                DatasetCommand::Remove(id) => {
                    if id == "__shutdown__" {
                        return;
                    }
                    if let Some(candidate) = candidates.remove(&id) {
                        if let Some(path) = candidate.original_path {
                            let _ = fs::remove_file(path);
                        }
                        if let Some(path) = candidate.prepared_path {
                            let _ = fs::remove_file(path);
                        }
                        let dir = root.join("datasets/discovery").join(safe_fs_id(&id));
                        let _ = fs::remove_dir_all(dir);
                        let _ = events.send(DatasetEvent::Removed(id));
                        persist_candidates(&root, candidates.values());
                    }
                }
            }
        }

        if !paused && !stopped {
            if let Some((id, allow_large, auto_prepare)) = download_queue.pop_front() {
                if let Some(candidate) = candidates.get_mut(&id) {
                    match download_candidate(&root, candidate.clone(), allow_large, &events) {
                        Ok(updated) => {
                            candidates.insert(id.clone(), updated.clone());
                            let _ = events.send(DatasetEvent::Candidate(updated.clone()));

                            if auto_prepare {
                                match prepare_candidate(&root, &updated, &events) {
                                    Ok(mut ready) => {
                                        candidates.insert(id.clone(), ready.clone());
                                        let _ = events.send(DatasetEvent::Ready(ready.clone()));
                                        ready.state = DatasetState::UsedInTraining;
                                        candidates.insert(id.clone(), ready.clone());
                                        let _ = events.send(DatasetEvent::UsedForTraining(ready));
                                    }
                                    Err(error) => {
                                        let mut failed = updated;
                                        failed.state = DatasetState::Failed;
                                        failed.error = Some(error.clone());
                                        candidates.insert(id.clone(), failed.clone());
                                        let _ = events.send(DatasetEvent::Candidate(failed));
                                        let _ = events.send(DatasetEvent::Error {
                                            id: Some(id.clone()),
                                            message: error,
                                        });
                                    }
                                }
                            }
                            persist_candidates(&root, candidates.values());
                        }
                    }
                    Err(error) => {
                        if error.starts_with("APPROVAL:") {
                            let size = error.trim_start_matches("APPROVAL:").parse::<u64>().unwrap_or(0);
                            if let Some(candidate) = candidates.get_mut(&id) {
                                candidate.state = DatasetState::AwaitingApproval;
                                let _ = events.send(DatasetEvent::Candidate(candidate.clone()));
                                let _ = events.send(DatasetEvent::DownloadNeedsApproval {
                                    id: id.clone(),
                                    size_bytes: size,
                                    limit_bytes: DEFAULT_MAX_DOWNLOAD_BYTES,
                                });
                            }
                        } else {
                            let mut failed = candidate.clone();
                            failed.state = DatasetState::Failed;
                            failed.error = Some(error.clone());
                            candidates.insert(id.clone(), failed.clone());
                            let _ = events.send(DatasetEvent::Candidate(failed));
                            let _ = events.send(DatasetEvent::Error { id: Some(id.clone()), message: error });
                        }
                    }
                }
            }
        }

        thread::sleep(Duration::from_millis(50));
    }
}

fn download_candidate(
    root: &Path,
    mut candidate: DatasetCandidate,
    allow_large: bool,
    events: &Sender<DatasetEvent>,
) -> Result<DatasetCandidate, String> {
    let url = candidate.download_url.clone().ok_or_else(|| "Dataset has no public downloadable data file.".to_string())?;
    if !safe_http_url(&url) {
        return Err("Dataset download URL is not http/https.".into());
    }
    let path_hint = Url::parse(&url)
        .ok()
        .and_then(|v| v.path_segments().and_then(|mut s| s.next_back().map(str::to_string)))
        .unwrap_or_else(|| "dataset.data".into());
    if !allowed_download_name(&path_hint) {
        return Err(format!("Unsupported or unsafe dataset file: {path_hint}"));
    }
    let extension = Path::new(&path_hint).extension().and_then(|v| v.to_str()).unwrap_or("data").to_ascii_lowercase();
    let dir = root.join("datasets/discovery").join(safe_fs_id(&candidate.id));
    fs::create_dir_all(&dir).map_err(|e| format!("create dataset directory: {e}"))?;
    let output = dir.join(format!("original.{extension}"));
    if output.exists() && output.metadata().map(|m| m.len()).unwrap_or(0) > 0 {
        candidate.original_path = Some(output);
        candidate.state = DatasetState::Downloaded;
        return Ok(candidate);
    }
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(60)).user_agent(USER_AGENT).build();
    let response = agent.get(&url).set("Accept", "*/*").call().map_err(|e| format!("dataset download request: {e}"))?;
    let total = response.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).or(candidate.size_bytes);
    if let Some(size) = total {
        candidate.size_bytes = Some(size);
        candidate.size_label = human_size(size);
        if size > ABSOLUTE_MAX_DOWNLOAD_BYTES {
            return Err(format!("Dataset is {} and exceeds the absolute safety limit of {}.", human_size(size), human_size(ABSOLUTE_MAX_DOWNLOAD_BYTES)));
        }
        if size > DEFAULT_MAX_DOWNLOAD_BYTES && !allow_large {
            return Err(format!("APPROVAL:{size}"));
        }
    }
    let mut reader = response.into_reader().take(ABSOLUTE_MAX_DOWNLOAD_BYTES + 1);
    let mut file = File::create(&output).map_err(|e| format!("create dataset file: {e}"))?;
    let mut buffer = [0u8; 64 * 1024];
    let mut downloaded = 0u64;
    loop {
        let read = reader.read(&mut buffer).map_err(|e| format!("read dataset download: {e}"))?;
        if read == 0 {
            break;
        }
        downloaded += read as u64;
        if downloaded > ABSOLUTE_MAX_DOWNLOAD_BYTES {
            let _ = fs::remove_file(&output);
            return Err("Dataset download exceeded the absolute safety limit.".into());
        }
        file.write_all(&buffer[..read]).map_err(|e| format!("write dataset: {e}"))?;
        let _ = events.send(DatasetEvent::DownloadProgress {
            id: candidate.id.clone(),
            downloaded_bytes: downloaded,
            total_bytes: total,
        });
    }
    file.flush().map_err(|e| format!("flush dataset: {e}"))?;
    if downloaded == 0 {
        let _ = fs::remove_file(&output);
        return Err("Dataset download returned an empty file.".into());
    }
    candidate.original_path = Some(output);
    candidate.size_bytes = Some(downloaded);
    candidate.size_label = human_size(downloaded);
    candidate.state = DatasetState::Downloaded;
    candidate.error = None;
    Ok(candidate)
}

fn prepare_candidate(
    root: &Path,
    candidate: &DatasetCandidate,
    _events: &Sender<DatasetEvent>,
) -> Result<DatasetCandidate, String> {
    let input = candidate.original_path.clone().ok_or("Download the dataset first.")?;
    let dir = root.join("datasets/discovery").join(safe_fs_id(&candidate.id));
    fs::create_dir_all(&dir).map_err(|e| format!("prepare dataset directory: {e}"))?;
    let output = dir.join("prepared.aicorpus");
    let format = input.extension().and_then(|v| v.to_str()).unwrap_or_default().to_ascii_lowercase();
    let report = DatasetNormalizer::normalize(&input, &format, &output)?;
    if report.samples == 0 {
        return Err("Unsupported dataset structure: no usable text records were found.".into());
    }
    let mut updated = candidate.clone();
    updated.prepared_path = Some(output);
    updated.state = DatasetState::Ready;
    updated.error = None;
    updated.description = format!("{} • {} samples • {}", updated.description, report.samples, human_size(report.output_bytes));
    Ok(updated)
}

#[derive(Clone, Debug, Default)]
pub struct NormalizationReport {
    pub samples: u64,
    pub rejected: u64,
    pub output_bytes: u64,
}

pub struct DatasetNormalizer;
impl DatasetNormalizer {
    pub fn normalize(input: &Path, format: &str, output: &Path) -> Result<NormalizationReport, String> {
        match format {
            "txt" | "text" | "jsonl" | "json" | "csv" => normalize_line_or_structured(input, format, output),
            "bz2" => normalize_wikimedia_bz2(input, output),
            "parquet" => Err("Parquet is indexed by discovery but its decoder is intentionally not bundled in this lightweight client.".into()),
            "zip" | "gz" | "tar" => Err("Archive detected. The selected archive format is not enabled by the lightweight normalizer.".into()),
            other => Err(format!("Unsupported dataset format: {other}")),
        }
    }
}

fn normalize_line_or_structured(input: &Path, format: &str, output: &Path) -> Result<NormalizationReport, String> {
    let file = File::open(input).map_err(|e| format!("open dataset: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut writer = File::create(output).map_err(|e| format!("create AICORPUS: {e}"))?;
    writeln!(writer, "# AiCorpus v1").map_err(|e| format!("write AICORPUS header: {e}"))?;
    let mut report = NormalizationReport::default();
    let mut line = String::new();
    match format {
        "txt" | "text" => {
            while reader.read_line(&mut line).map_err(|e| format!("read dataset: {e}"))? != 0 {
                if append_normalized(&mut writer, &line)? {
                    report.samples += 1;
                } else {
                    report.rejected += 1;
                }
                line.clear();
            }
        }
        "jsonl" => {
            while reader.read_line(&mut line).map_err(|e| format!("read JSONL: {e}"))? != 0 {
                if let Ok(value) = serde_json::from_str::<Value>(&line) {
                    if let Some(text) = extract_text_value(&value) {
                        if append_normalized(&mut writer, &text)? {
                            report.samples += 1;
                        } else {
                            report.rejected += 1;
                        }
                    } else {
                        report.rejected += 1;
                    }
                } else {
                    report.rejected += 1;
                }
                line.clear();
            }
        }
        "csv" => {
            let mut header = String::new();
            reader.read_line(&mut header).map_err(|e| format!("read CSV header: {e}"))?;
            let headers = parse_csv_record(&header);
            while reader.read_line(&mut line).map_err(|e| format!("read CSV: {e}"))? != 0 {
                let values = parse_csv_record(&line);
                let mut map = HashMap::new();
                for (index, value) in values.iter().enumerate() {
                    if let Some(key) = headers.get(index) {
                        map.insert(key.to_ascii_lowercase(), value.clone());
                    }
                }
                let text = extract_text_from_map(&map);
                if append_normalized(&mut writer, &text)? {
                    report.samples += 1;
                } else {
                    report.rejected += 1;
                }
                line.clear();
            }
        }
        "json" => {
            let mut body = String::new();
            reader.read_to_string(&mut body).map_err(|e| format!("read JSON: {e}"))?;
            let value: Value = serde_json::from_str(&body).map_err(|e| format!("parse JSON: {e}"))?;
            let mut records = Vec::new();
            collect_json_records(&value, &mut records);
            for text in records {
                if append_normalized(&mut writer, &text)? {
                    report.samples += 1;
                } else {
                    report.rejected += 1;
                }
            }
        }
        _ => unreachable!(),
    }
    writer.flush().map_err(|e| format!("flush AICORPUS: {e}"))?;
    report.output_bytes = output.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(report)
}

fn normalize_wikimedia_bz2(input: &Path, output: &Path) -> Result<NormalizationReport, String> {
    let file = File::open(input).map_err(|e| format!("open Wikimedia dump: {e}"))?;
    let decoder = bzip2::read::BzDecoder::new(BufReader::new(file));
    let mut reader = BufReader::new(decoder);
    let mut writer = File::create(output).map_err(|e| format!("create AICORPUS: {e}"))?;
    writeln!(writer, "# AiCorpus v1").map_err(|e| e.to_string())?;
    let mut body = String::new();
    let mut report = NormalizationReport::default();
    let mut in_text = false;
    let mut text = String::new();
    let mut bytes_read = 0u64;
    loop {
        let mut buf = [0u8; 64 * 1024];
        let n = reader.read(&mut buf).map_err(|e| format!("read Wikimedia dump: {e}"))?;
        if n == 0 { break; }
        bytes_read += n as u64;
        body.push_str(&String::from_utf8_lossy(&buf));
        let mut start = 0usize;
        while let Some(rel) = body[start..].find(if in_text { "</text>" } else { "<text" }) {
            let pos = start + rel;
            if !in_text {
                if let Some(end) = body[pos..].find('>') {
                    start = pos + end + 1;
                    in_text = true;
                } else {
                    break;
                }
            } else {
                if append_normalized(&mut writer, &body[start..pos])? { report.samples += 1; } else { report.rejected += 1; }
                start = pos + "</text>".len();
                text.clear();
                in_text = false;
            }
        }
        if start > 0 {
            body = body[start..].to_string();
        }
        if body.len() > 4 * 1024 * 1024 {
            body.truncate(512 * 1024);
        }
        if bytes_read > 2 * 1024 * 1024 * 1024 {
            return Err("Wikimedia decompressed data exceeded the safety limit.".into());
        }
    }
    let _ = text;
    writer.flush().map_err(|e| format!("flush AICORPUS: {e}"))?;
    report.output_bytes = output.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(report)
}

fn append_normalized(writer: &mut File, raw: &str) -> Result<bool, String> {
    let text = collapse_text(raw);
    if text.len() < 32 {
        return Ok(false);
    }
    writer
        .write_all(text.as_bytes())
        .and_then(|_| writer.write_all(b"\n"))
        .map_err(|e| format!("write normalized corpus: {e}"))?;
    Ok(true)
}

fn extract_text_value(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Object(map) => {
            for key in ["text", "content", "body", "article", "document", "description"] {
                if let Some(text) = map.get(key).and_then(Value::as_str) {
                    return Some(text.to_string());
                }
            }
            for (a, b) in [("question", "answer"), ("prompt", "response"), ("user", "assistant")] {
                if let (Some(left), Some(right)) = (map.get(a).and_then(Value::as_str), map.get(b).and_then(Value::as_str)) {
                    return Some(format!("{left}\n{right}"));
                }
            }
            for child in map.values() {
                if let Some(text) = extract_text_value(child) {
                    return Some(text);
                }
            }
            None
        }
        Value::Array(items) => items.iter().find_map(extract_text_value),
        _ => None,
    }
}

fn extract_text_from_map(map: &HashMap<String, String>) -> String {
    for key in ["text", "content", "body", "article", "document", "description"] {
        if let Some(value) = map.get(key) {
            return value.clone();
        }
    }
    for (a, b) in [("question", "answer"), ("prompt", "response"), ("user", "assistant")] {
        if let (Some(left), Some(right)) = (map.get(a), map.get(b)) {
            return format!("{left}\n{right}");
        }
    }
    String::new()
}

fn collect_json_records(value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Array(items) => for item in items { collect_json_records(item, out); },
        Value::Object(map) => {
            if let Some(text) = extract_text_value(value) {
                out.push(text);
            } else {
                for child in map.values() { collect_json_records(child, out); }
            }
        }
        Value::String(text) if text.len() >= 32 => out.push(text.clone()),
        _ => {}
    }
}

fn parse_csv_record(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut chars = line.trim_end_matches(['\r', '\n']).chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                current.push('"');
                let _ = chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                out.push(current.clone());
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    out.push(current);
    out
}

fn preview_candidate(candidate: &DatasetCandidate) -> Result<Vec<String>, String> {
    let url = candidate
        .download_url
        .as_deref()
        .ok_or("Dataset has no public preview/download URL.")?;
    if candidate.format.eq_ignore_ascii_case("XML+BZip2") || url.ends_with(".bz2") {
        return Err("Wikimedia preview is not materialized as a full dump; download and prepare it first.".into());
    }
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(20)).user_agent(USER_AGENT).build();
    let response = agent.get(url).set("Range", &format!("bytes=0-{}", MAX_PREVIEW_BYTES - 1)).call().map_err(|e| format!("preview request: {e}"))?;
    let mut bytes = Vec::new();
    response.into_reader().take(MAX_PREVIEW_BYTES as u64).read_to_end(&mut bytes).map_err(|e| format!("read preview: {e}"))?;
    let text = String::from_utf8_lossy(&bytes).into_owned();
    let mut out = Vec::new();
    if candidate.format.eq_ignore_ascii_case("JSONL") {
        for line in text.lines().take(3) {
            if let Ok(value) = serde_json::from_str::<Value>(line) {
                if let Some(sample) = extract_text_value(&value) { out.push(collapse_text(&sample)); }
            }
        }
    } else {
        for line in text.lines().filter(|v| !v.trim().is_empty()).take(3) {
            out.push(collapse_text(line));
        }
    }
    Ok(out)
}

fn load_cached_candidates(root: &Path) -> HashMap<String, DatasetCandidate> {
    let mut map = HashMap::new();
    let dir = root.join("datasets/discovery");
    let Ok(entries) = fs::read_dir(dir) else { return map; };
    for entry in entries.flatten() {
        let path = entry.path().join("manifest.json");
        let Ok(text) = fs::read_to_string(path) else { continue; };
        let Ok(candidate) = serde_json::from_str::<DatasetCandidate>(&text) else { continue; };
        map.insert(candidate.id.clone(), candidate);
    }
    map
}

fn persist_candidates<'a>(root: &Path, candidates: impl Iterator<Item = &'a DatasetCandidate>) {
    for candidate in candidates {
        let dir = root.join("datasets/discovery").join(safe_fs_id(&candidate.id));
        if fs::create_dir_all(&dir).is_err() { continue; }
        let path = dir.join("manifest.json.tmp");
        if let Ok(bytes) = serde_json::to_vec_pretty(candidate) {
            if fs::write(&path, bytes).is_ok() {
                let _ = fs::rename(path, dir.join("manifest.json"));
            }
        }
    }
}

fn get_json(url: &str) -> Result<Value, String> {
    let text = get_text(url, 8 * 1024 * 1024)?;
    serde_json::from_str(&text).map_err(|e| format!("JSON from {url}: {e}"))
}
fn get_text(url: &str, max: usize) -> Result<String, String> {
    if !safe_http_url(url) { return Err("non-http dataset source URL".into()); }
    let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(30)).user_agent(USER_AGENT).build();
    let response = agent.get(url).set("Accept", "application/json,text/html;q=0.9,*/*;q=0.2").call().map_err(|e| format!("dataset source request: {e}"))?;
    let mut bytes = Vec::new();
    response.into_reader().take((max + 1) as u64).read_to_end(&mut bytes).map_err(|e| format!("read dataset source: {e}"))?;
    if bytes.len() > max { return Err(format!("dataset source response exceeds {}.", human_size(max as u64))); }
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn join_query(filters: &DatasetSearchFilters) -> String {
    [filters.query.as_str(), filters.topic.as_str(), filters.kind.as_str()]
        .iter()
        .copied()
        .filter(|v| !v.trim().is_empty() && !v.eq_ignore_ascii_case("Any"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn form_encode(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn safe_http_url(value: &str) -> bool {
    Url::parse(value)
        .map(|url| matches!(url.scheme(), "http" | "https") && url.host_str().is_some())
        .unwrap_or(false)
}

fn allowed_download_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    ["txt", "text", "json", "jsonl", "csv", "parquet", "bz2", "zip", "gz", "tar"]
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

fn allowed_data_extension(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    ["txt", "text", "json", "jsonl", "csv", "parquet"]
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

fn data_file_rank(path: &str) -> usize {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".jsonl") { 0 } else if lower.ends_with(".txt") || lower.ends_with(".text") { 1 } else if lower.ends_with(".csv") { 2 } else if lower.ends_with(".json") { 3 } else if lower.ends_with(".parquet") { 4 } else { 9 }
}

fn display_format(path: &str) -> String {
    if path.ends_with(".jsonl") { "JSONL".into() }
    else if path.ends_with(".json") { "JSON".into() }
    else if path.ends_with(".csv") { "CSV".into() }
    else if path.ends_with(".txt") || path.ends_with(".text") { "TXT".into() }
    else if path.ends_with(".parquet") { "Parquet".into() }
    else { "Unknown".into() }
}

fn infer_kind_from_path(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.contains("conversation") || lower.contains("dialog") || lower.contains("chat") { "Conversation" }
    else if lower.contains("qa") || lower.contains("question") || lower.contains("answer") { "Knowledge" }
    else { "Text" }
}

fn infer_kind(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if ["conversation", "chat", "dialog", "assistant"].iter().any(|v| lower.contains(v)) { "Conversation".into() }
    else if ["question", "answer", "knowledge"].iter().any(|v| lower.contains(v)) { "Knowledge".into() }
    else { "Text".into() }
}

fn infer_topic(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    for (name, terms) in [
        ("Technology", &["technology", "computer", "software", "ai"][..]),
        ("Science", &["science", "research", "physics", "biology"][..]),
        ("Knowledge", &["encyclopedia", "knowledge", "wiki", "reference"][..]),
        ("Conversation", &["conversation", "chat", "dialog"][..]),
    ] {
        if terms.iter().any(|v| lower.contains(v)) { return name.into(); }
    }
    "General text".into()
}

fn infer_language(text: &str) -> String {
    let lower = text.to_ascii_lowercase();
    if ["russian", "русский", "ru"].iter().any(|v| lower.contains(v)) { "Russian".into() }
    else if ["ukrainian", "україн", "uk"].iter().any(|v| lower.contains(v)) { "Ukrainian".into() }
    else if ["english", "en"].iter().any(|v| lower.contains(v)) { "English".into() }
    else { "Unknown".into() }
}

fn size_matches(size_bytes: Option<u64>, wanted: &str) -> bool {
    match wanted.to_ascii_lowercase().as_str() {
        "any" => true,
        "small" => size_bytes.map(|v| v <= 128 * 1024 * 1024).unwrap_or(true),
        "medium" => size_bytes.map(|v| v > 128 * 1024 * 1024 && v <= 1024 * 1024 * 1024).unwrap_or(true),
        "large" => size_bytes.map(|v| v > 1024 * 1024 * 1024).unwrap_or(true),
        _ => true,
    }
}

fn kind_matches(wanted: &str, detected: &str) -> bool {
    wanted.eq_ignore_ascii_case("Any") || detected.eq_ignore_ascii_case(wanted)
}

fn topic_matches(wanted: &str, detected: &str) -> bool {
    wanted.eq_ignore_ascii_case("Any")
        || wanted.eq_ignore_ascii_case("General text")
        || detected.eq_ignore_ascii_case(wanted)
}

fn language_matches(wanted: &str, tags: &str, description: &str) -> bool {
    if wanted.eq_ignore_ascii_case("All") || wanted.eq_ignore_ascii_case("Any") { return true; }
    let haystack = format!("{} {}", tags, description).to_ascii_lowercase();
    match wanted.to_ascii_lowercase().as_str() {
        "english" => haystack.contains("english") || haystack.contains("en"),
        "russian" => haystack.contains("russian") || haystack.contains("ru"),
        "ukrainian" => haystack.contains("ukrainian") || haystack.contains("uk"),
        _ => true,
    }
}

fn quality_for_metadata(description: &str, popularity: u64, tags: &str) -> f32 {
    let popularity_score = ((popularity as f64).ln_1p() / 12.0).clamp(0.0, 1.0) as f32;
    let structured = if tags.is_empty() { 0.55 } else { 0.70 };
    (0.55 * structured + 0.45 * popularity_score.max(0.20) + if description.len() > 40 { 0.05 } else { 0.0 }).clamp(0.0, 1.0)
}

fn collapse_text(input: &str) -> String {
    input
        .replace("&nbsp;", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

fn human_size(bytes: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut index = 0usize;
    while value >= 1024.0 && index + 1 < units.len() {
        value /= 1024.0;
        index += 1;
    }
    if index == 0 { format!("{bytes} {}", units[index]) } else { format!("{value:.1} {}", units[index]) }
}

fn safe_fs_id(id: &str) -> String {
    id.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
        .collect::<String>()
        .chars()
        .take(160)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_parser_handles_quotes() {
        let row = parse_csv_record("\"hello, world\",answer");
        assert_eq!(row, vec!["hello, world", "answer"]);
    }

    #[test]
    fn candidate_file_names_are_safe() {
        assert!(allowed_download_name("data.jsonl"));
        assert!(allowed_download_name("dump.xml.bz2"));
        assert!(!allowed_download_name("payload.exe"));
    }

    #[test]
    fn normalizes_text() {
        assert_eq!(collapse_text("  hello   world "), "hello world");
    }
}
