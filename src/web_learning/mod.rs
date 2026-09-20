#![allow(clippy::manual_is_ascii_check, clippy::needless_range_loop, unused_assignments)]
use encoding_rs::Encoding;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use ureq::Agent;
use url::Url;

const CORPUS_MAGIC: &[u8; 8] = b"AICORPUS";
const CORPUS_VERSION: u32 = 1;
const USER_AGENT: &str = "AiNet-WebLearner/1.0";
const DEFAULT_MAX_BODY_BYTES: usize = 2 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourcePriority {
    High,
    Normal,
    Low,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebStatus { Stopped, Starting, Running, Pausing, Paused, Stopping, Offline, Error }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceRecord {
    pub id: String,
    pub url: String,
    pub domain: String,
    pub feed_url: Option<String>,
    pub language: Option<String>,
    pub topic: Option<String>,
    pub country: Option<String>,
    pub priority: SourcePriority,
    pub enabled: bool,
    pub last_scan: Option<u64>,
    pub last_success: Option<u64>,
    pub error_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebSettings {
    pub languages: Vec<String>,
    pub topic: Option<String>,
    pub country: Option<String>,
    pub freshness_hours: u64,
    pub scan_interval_secs: u64,
    pub maximum_concurrent_domains: usize,
    pub maximum_concurrent_requests: usize,
    pub per_domain_delay_ms: u64,
    pub retry_count: usize,
    pub max_pages_per_scan: usize,
    pub max_body_bytes: usize,
    pub autonomous: bool,
    pub tokenizer_path: Option<PathBuf>,
    pub replay_ratio_percent: u8,
    pub retention_max_articles: usize,
}
impl Default for WebSettings {
    fn default() -> Self {
        Self {
            languages: vec!["English".into()],
            topic: None,
            country: None,
            freshness_hours: 24,
            scan_interval_secs: 900,
            maximum_concurrent_domains: 2,
            maximum_concurrent_requests: 2,
            per_domain_delay_ms: 1000,
            retry_count: 2,
            max_pages_per_scan: 24,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            autonomous: false,
            tokenizer_path: None,
            replay_ratio_percent: 20,
            retention_max_articles: 50_000,
        }
    }
}
impl WebSettings {
    pub fn normalize(&mut self) {
        self.scan_interval_secs = self.scan_interval_secs.max(15);
        self.maximum_concurrent_domains = self.maximum_concurrent_domains.clamp(1, 4);
        self.maximum_concurrent_requests = self.maximum_concurrent_requests.clamp(1, 4);
        self.per_domain_delay_ms = self.per_domain_delay_ms.max(250);
        self.retry_count = self.retry_count.min(5);
        self.max_pages_per_scan = self.max_pages_per_scan.clamp(1, 200);
        self.max_body_bytes = self.max_body_bytes.clamp(64 * 1024, 16 * 1024 * 1024);
        self.replay_ratio_percent = self.replay_ratio_percent.min(80);
        self.retention_max_articles = self.retention_max_articles.max(100);
        if self.languages.is_empty() { self.languages.push("English".into()); }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WebStats {
    pub sources: u64,
    pub active_sources: u64,
    pub pages_scanned: u64,
    pub pages_accepted: u64,
    pub pages_rejected: u64,
    pub articles_collected: u64,
    pub duplicates_skipped: u64,
    pub tokens_queued: u64,
    pub training_queue: u64,
    pub last_update: Option<u64>,
    pub current_source: Option<String>,
    pub current_url: Option<String>,
}

#[derive(Clone, Debug)]
pub enum WebEvent {
    Status(WebStatus),
    Stats(WebStats),
    ArticleAccepted(ArticleRecord),
    Error { source_id: Option<String>, error: String },
    Offline(String),
    Sources(Vec<SourceRecord>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArticleRecord {
    pub id: String,
    pub url: String,
    pub canonical_url: String,
    pub domain: String,
    pub title: String,
    pub published_at: Option<String>,
    pub fetched_at: u64,
    pub language: String,
    pub text: String,
    pub content_hash: u64,
    pub source_id: String,
    pub word_count: usize,
    pub token_count: usize,
    pub quality_score: f32,
}

#[derive(Clone, Debug)]
struct FetchedPage {
    final_url: String,
    content_type: String,
    bytes: Vec<u8>,
    etag: Option<String>,
    last_modified: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct ParsedDocument {
    title: String,
    published_at: Option<String>,
    text: String,
    alternate_feeds: Vec<String>,
    links: Vec<String>,
}

#[derive(Clone, Debug)]
struct DiscoveredItem {
    url: String,
    title: Option<String>,
    published_at: Option<String>,
    content_hint: Option<String>,
}

#[derive(Clone, Debug)]
struct ScanResult {
    source: SourceRecord,
    pages_scanned: u64,
    rejected: u64,
    articles: Vec<ArticleRecord>,
    errors: Vec<String>,
}

#[derive(Clone)]
struct DomainLimiter { next_allowed: Arc<Mutex<HashMap<String, Instant>>> }
impl DomainLimiter {
    fn new() -> Self { Self { next_allowed: Arc::new(Mutex::new(HashMap::new())) } }
    fn wait_turn(&self, domain: &str, delay: Duration) {
        let now = Instant::now();
        let target = {
            let mut map = self.next_allowed.lock().unwrap_or_else(|e| e.into_inner());
            let target = map.get(domain).copied().unwrap_or(now);
            map.insert(domain.to_string(), target.max(now) + delay);
            target
        };
        if target > now { thread::sleep(target - now); }
    }
}

pub struct WebFetcher {
    agent: Agent,
    max_body_bytes: usize,
}
impl WebFetcher {
    pub fn new(max_body_bytes: usize) -> Result<Self, String> {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(20))
            .user_agent(USER_AGENT)
            .build();
        Ok(Self { agent, max_body_bytes })
    }

    fn fetch(&self, url: &str, etag: Option<&str>, last_modified: Option<&str>) -> Result<Option<FetchedPage>, String> {
        let mut request = self.agent.get(url)
            .set("Accept", "text/html,application/rss+xml,application/atom+xml,application/xml,application/json;q=0.8,text/plain;q=0.5")
            .set("Accept-Encoding", "gzip")
            .set("User-Agent", USER_AGENT);
        if let Some(value) = etag { request = request.set("If-None-Match", value); }
        if let Some(value) = last_modified { request = request.set("If-Modified-Since", value); }

        match request.call() {
            Ok(response) => {
                let length = response.header("Content-Length").and_then(|v| v.parse::<usize>().ok());
                if length.is_some_and(|v| v > self.max_body_bytes) {
                    return Err(format!("response exceeds {} byte limit", self.max_body_bytes));
                }
                let content_type = response.header("Content-Type").unwrap_or("text/plain").to_string();
                let final_url = response.get_url().to_string();
                let etag = response.header("ETag").map(str::to_string);
                let last_modified = response.header("Last-Modified").map(str::to_string);
                let mut bytes = Vec::new();
                response.into_reader().take((self.max_body_bytes + 1) as u64)
                    .read_to_end(&mut bytes)
                    .map_err(|e| format!("read response: {e}"))?;
                if bytes.len() > self.max_body_bytes {
                    return Err("response exceeded streaming size limit".into());
                }
                Ok(Some(FetchedPage { final_url, content_type, bytes, etag, last_modified }))
            }
            Err(ureq::Error::Status(304, _)) => Ok(None),
            Err(ureq::Error::Status(code, _)) => Err(format!("HTTP status {code}")),
            Err(error) => Err(format!("HTTP request failed: {error}")),
        }
    }
}

pub struct ContentCleaner;
impl ContentCleaner {
    fn clean(html: &str) -> ParsedDocument {
        let mut doc = ParsedDocument::default();
        let mut output = String::new();
        let mut skip_depth = 0usize;
        let mut in_title = false;
        let mut i = 0usize;

        while i < html.len() {
            if html.as_bytes()[i] == b'<' {
                if let Some(end_rel) = html[i..].find('>') {
                    let end = i + end_rel;
                    let raw = &html[i + 1..end];
                    let (name, closing, attrs) = parse_tag(raw);
                    let tag = name.to_ascii_lowercase();

                    if tag == "title" { in_title = !closing; }
                    if !closing && matches!(tag.as_str(), "script"|"style"|"noscript"|"svg"|"nav"|"footer"|"header"|"aside"|"form"|"dialog") {
                        skip_depth += 1;
                    } else if closing && matches!(tag.as_str(), "script"|"style"|"noscript"|"svg"|"nav"|"footer"|"header"|"aside"|"form"|"dialog") {
                        skip_depth = skip_depth.saturating_sub(1);
                    }
                    if !closing && tag == "meta" {
                        if let Some((key, value)) = meta_pair(&attrs) {
                            match key.to_ascii_lowercase().as_str() {
                                "article:published_time" | "date" | "pubdate" => doc.published_at = Some(value),
                                _ => {}
                            }
                        }
                    }
                    if !closing && tag == "time" && doc.published_at.is_none() {
                        if let Some(value) = attr_value(&attrs, "datetime") { doc.published_at = Some(value.to_string()); }
                    }
                    if !closing && (tag == "a" || tag == "link") {
                        if let Some(href) = attr_value(&attrs, "href") {
                            let href = href.to_string();
                            if tag == "link" {
                                let rel = attr_value(&attrs, "rel").unwrap_or_default().to_ascii_lowercase();
                                let kind = attr_value(&attrs, "type").unwrap_or_default().to_ascii_lowercase();
                                if rel.split_whitespace().any(|v| v == "alternate") && (kind.contains("rss") || kind.contains("atom") || kind.contains("xml")) {
                                    doc.alternate_feeds.push(href.clone());
                                }
                            }
                            doc.links.push(href);
                        }
                    }
                    i = end + 1;
                    continue;
                }
            }

            if skip_depth == 0 {
                if let Some(next) = html[i..].find('<') {
                    let raw = decode_entities(&html[i..i + next]);
                    if in_title { append_text(&mut doc.title, &raw); } else { append_text(&mut output, &raw); }
                    i += next;
                } else {
                    let raw = decode_entities(&html[i..]);
                    if in_title { append_text(&mut doc.title, &raw); } else { append_text(&mut output, &raw); }
                    break;
                }
            } else if let Some(next) = html[i..].find('<') {
                i += next;
            } else { break; }
        }
        doc.text = output;
        doc.title = doc.title.trim().to_string();
        doc
    }
}

pub struct LanguageDetector;
impl LanguageDetector {
    pub fn detect(text: &str) -> String {
        let mut latin = 0usize;
        let mut cyrillic = 0usize;
        let mut uk = 0usize;
        let mut letters = 0usize;
        for ch in text.chars() {
            if ch.is_alphabetic() {
                letters += 1;
                let lower = ch.to_ascii_lowercase();
                if lower.is_ascii_lowercase() { latin += 1; }
                if ('а'..='я').contains(&lower) || "ёіїєґ".contains(lower) { cyrillic += 1; }
                if "іїєґ".contains(lower) { uk += 1; }
            }
        }
        if letters == 0 { return "Unknown".into(); }
        if uk >= 2 && cyrillic > 5 { return "Ukrainian".into(); }
        if cyrillic as f32 / letters as f32 > 0.35 { return "Russian".into(); }
        if latin as f32 / letters as f32 > 0.45 { return "English".into(); }
        "Other".into()
    }
}

pub struct QualityFilter;
impl QualityFilter {
    pub fn score(text: &str) -> f32 {
        let chars = text.chars().count();
        let words = text.split_whitespace().count();
        if chars < 400 || words < 60 || chars > 200_000 { return 0.0; }
        let alpha = text.chars().filter(|c| c.is_alphabetic()).count();
        let alpha_ratio = alpha as f32 / chars.max(1) as f32;
        let links = text.matches("http://").count() + text.matches("https://").count();
        let repeat = repeated_word_ratio(text);
        let lower = text.to_ascii_lowercase();
        let boilerplate = ["accept cookies","cookie policy","privacy policy","subscribe to our newsletter","sign in to continue"]
            .iter().filter(|v| lower.contains(**v)).count();
        (0.5 * alpha_ratio + 0.35 * (1.0 - repeat).max(0.0)
            + 0.15 * (1.0 - (links as f32 / words.max(1) as f32).min(0.5))
            - boilerplate as f32 * 0.12).clamp(0.0, 1.0)
    }
    pub fn accept(text: &str) -> bool { Self::score(text) >= 0.42 }
}

pub struct TopicClassifier;
impl TopicClassifier {
    pub fn classify(text: &str) -> String {
        let lower = text.to_ascii_lowercase();
        let topics = [
            ("Technology", ["software","computer","chip","ai","technology","internet","cyber"]),
            ("Science", ["science","research","physics","biology","space","climate","study"]),
            ("World", ["world","international","government","president","country","war","diplomatic"]),
            ("Business", ["business","market","economy","finance","company","stock","trade"]),
            ("Culture", ["culture","film","music","book","art","museum","festival"]),
            ("Sports", ["sports","football","basketball","tennis","league","match","olympic"]),
            ("Gaming", ["game","gaming","xbox","playstation","nintendo","steam","esports"]),
        ];
        let mut best = ("Other", 0usize);
        for (topic, words) in topics {
            let score = words.iter().filter(|v| lower.contains(*v)).count();
            if score > best.1 { best = (topic, score); }
        }
        best.0.into()
    }
}

pub struct WebParser;
impl WebParser {
    fn decode(bytes: &[u8], content_type: &str) -> Result<String, String> {
        let charset = content_type.split(';').find_map(|part| {
            part.trim().strip_prefix("charset=").map(str::trim)
        }).unwrap_or("utf-8");
        if charset.eq_ignore_ascii_case("utf-8") || charset.eq_ignore_ascii_case("utf8") || charset.eq_ignore_ascii_case("us-ascii") {
            Ok(String::from_utf8_lossy(bytes).into_owned())
        } else if let Some(enc) = Encoding::for_label(charset.as_bytes()) {
            let (text, _, _) = enc.decode(bytes);
            Ok(text.into_owned())
        } else {
            Err(format!("unsupported encoding: {charset}"))
        }
    }
    fn is_feed(content_type: &str, body: &str) -> bool {
        let ct = content_type.to_ascii_lowercase();
        ct.contains("rss") || ct.contains("atom") || body.contains("<rss") || body.contains("<feed")
    }
    fn is_sitemap(body: &str) -> bool { body.contains("<urlset") || body.contains("<sitemapindex") }
    fn feed_items(body: &str) -> Vec<DiscoveredItem> {
        let tag = if body.contains("<entry") { "entry" } else { "item" };
        let mut out = Vec::new();
        for block in tag_blocks(body, tag) {
            let url = extract_tag_value(block, "link").or_else(|| extract_attr_value(block, "link", "href"));
            let Some(url) = url else { continue };
            let title = extract_tag_value(block, "title");
            let published_at = extract_tag_value(block, "published").or_else(|| extract_tag_value(block, "pubDate")).or_else(|| extract_tag_value(block, "updated"));
            let content_hint = extract_tag_value(block, "content").or_else(|| extract_tag_value(block, "description")).or_else(|| extract_tag_value(block, "summary"));
            out.push(DiscoveredItem { url: decode_entities(&url), title: title.map(|v| decode_entities(&v)), published_at: published_at.map(|v| decode_entities(&v)), content_hint: content_hint.map(|v| decode_entities(&v)) });
        }
        out
    }
    fn sitemap_links(body: &str) -> Vec<String> {
        tag_values(body, "loc").into_iter().map(|v| decode_entities(&v)).collect()
    }
    fn json_items(body: &str) -> Vec<DiscoveredItem> {
        let Ok(value) = serde_json::from_str::<Value>(body) else { return Vec::new() };
        let mut out = Vec::new();
        walk_json(&value, &mut out);
        out
    }
}

pub struct WebDiscovery;
impl WebDiscovery {
    fn scan_source(source: SourceRecord, settings: &WebSettings, limiter: DomainLimiter, root: &Path) -> ScanResult {
        let fetcher = match WebFetcher::new(settings.max_body_bytes) {
            Ok(v) => v,
            Err(error) => return ScanResult { source, pages_scanned: 0, rejected: 0, articles: Vec::new(), errors: vec![error] }
        };
        let store = match CorpusStore::open(root) {
            Ok(v) => v,
            Err(error) => return ScanResult { source, pages_scanned: 0, rejected: 0, articles: Vec::new(), errors: vec![error] }
        };
        let mut source = source;
        source.last_scan = Some(now_ms());
        let mut pages_scanned = 0u64;
        let mut rejected = 0u64;
        let mut errors = Vec::new();
        let mut candidates = Vec::new();

        if let Some(feed) = &source.feed_url {
            if robots_allowed(&fetcher, &limiter, feed, settings.per_domain_delay_ms).unwrap_or(false) {
                if let Ok(Some(page)) = fetch_cached(&fetcher, &store, feed) {
                    if let Ok(body) = WebParser::decode(&page.bytes, &page.content_type) {
                        if WebParser::is_feed(&page.content_type, &body) { candidates.extend(WebParser::feed_items(&body)); }
                    }
                }
            }
        }

        if candidates.is_empty() {
            let base = source.url.clone();
            if robots_allowed(&fetcher, &limiter, &base, settings.per_domain_delay_ms).unwrap_or(false) {
                if let Ok(Some(page)) = fetch_cached(&fetcher, &store, &base) {
                    if let Ok(body) = WebParser::decode(&page.bytes, &page.content_type) {
                        if WebParser::is_feed(&page.content_type, &body) {
                            candidates.extend(WebParser::feed_items(&body));
                        } else if WebParser::is_sitemap(&body) {
                            candidates.extend(WebParser::sitemap_links(&body).into_iter().map(|url| DiscoveredItem { url, title: None, published_at: None, content_hint: None }));
                        } else if page.content_type.to_ascii_lowercase().contains("json") || body.trim_start().starts_with('{') || body.trim_start().starts_with('[') {
                            candidates.extend(WebParser::json_items(&body));
                        } else {
                            let doc = ContentCleaner::clean(&body);
                            for feed in doc.alternate_feeds {
                                let absolute = absolute_url(&page.final_url, &feed);
                                if !robots_allowed(&fetcher, &limiter, &absolute, settings.per_domain_delay_ms).unwrap_or(false) { continue; }
                                if let Ok(Some(feed_page)) = fetch_cached(&fetcher, &store, &absolute) {
                                    if let Ok(feed_body) = WebParser::decode(&feed_page.bytes, &feed_page.content_type) {
                                        if WebParser::is_feed(&feed_page.content_type, &feed_body) { candidates.extend(WebParser::feed_items(&feed_body)); }
                                    }
                                }
                            }
                            if candidates.is_empty() {
                                for suffix in ["/sitemap.xml", "/rss.xml", "/feed.xml", "/atom.xml"] {
                                    let probe = absolute_url(&base, suffix);
                                    if !robots_allowed(&fetcher, &limiter, &probe, settings.per_domain_delay_ms).unwrap_or(false) { continue; }
                                    if let Ok(Some(probe_page)) = fetch_cached(&fetcher, &store, &probe) {
                                        if let Ok(probe_body) = WebParser::decode(&probe_page.bytes, &probe_page.content_type) {
                                            if WebParser::is_feed(&probe_page.content_type, &probe_body) { candidates.extend(WebParser::feed_items(&probe_body)); break; }
                                            if WebParser::is_sitemap(&probe_body) { candidates.extend(WebParser::sitemap_links(&probe_body).into_iter().map(|url| DiscoveredItem { url, title: None, published_at: None, content_hint: None })); break; }
                                        }
                                    }
                                }
                            }
                            if candidates.is_empty() {
                                for link in doc.links {
                                    let absolute = absolute_url(&page.final_url, &link);
                                    if is_http_url(&absolute) { candidates.push(DiscoveredItem { url: absolute, title: None, published_at: None, content_hint: None }); }
                                    if candidates.len() >= settings.max_pages_per_scan { break; }
                                }
                            }
                        }
                    }
                }
            } else {
                errors.push("robots.txt denied or could not be read".into());
            }
        }

        let mut seen = HashSet::new();
        candidates.retain(|item| seen.insert(canonicalize_url(&item.url)));
        candidates.truncate(settings.max_pages_per_scan);

        let mut articles = Vec::new();
        for candidate in candidates {
            if !is_http_url(&candidate.url) { continue; }
            if !language_source_allowed(&source, &settings.languages) { continue; }
            let domain = Url::parse(&candidate.url).ok().and_then(|v| v.host_str().map(str::to_string)).unwrap_or_else(|| source.domain.clone());
            limiter.wait_turn(&domain, Duration::from_millis(settings.per_domain_delay_ms));
            if !robots_allowed(&fetcher, &limiter, &candidate.url, settings.per_domain_delay_ms).unwrap_or(false) { rejected += 1; continue; }

            let mut page = None;
            let mut last_error = None;
            for attempt in 0..=settings.retry_count {
                let cache = store.cache_entry(&candidate.url).ok().flatten();
                match fetcher.fetch(&candidate.url, cache.as_ref().and_then(|v| v.etag.as_deref()), cache.as_ref().and_then(|v| v.last_modified.as_deref())) {
                    Ok(value) => { page = value; break; }
                    Err(error) => {
                        last_error = Some(error);
                        if attempt < settings.retry_count { thread::sleep(Duration::from_millis(250u64.saturating_mul(1u64 << attempt.min(5)))); }
                    }
                }
            }
            pages_scanned += 1;
            let Some(page) = page else {
                rejected += 1;
                if let Some(error) = last_error { errors.push(error); }
                continue;
            };
            let body = match WebParser::decode(&page.bytes, &page.content_type) {
                Ok(v) => v,
                Err(error) => { rejected += 1; errors.push(error); continue; }
            };
            let _ = store.record_page_cache(&candidate.url, page.etag.as_deref(), page.last_modified.as_deref(), fnv1a64(&page.bytes));
            if WebParser::is_feed(&page.content_type, &body) {
                for item in WebParser::feed_items(&body).into_iter().take(settings.max_pages_per_scan) {
                    let Some(hint) = item.content_hint else { continue };
                    let text = strip_html(&hint);
                    if !QualityFilter::accept(&text) { continue; }
                    let article = make_article(&item.url, item.title.as_deref().unwrap_or("Untitled"), item.published_at, &text, &source.id);
                    if article_is_acceptable(&article, settings) { articles.push(article); }
                }
                continue;
            }
            let doc = ContentCleaner::clean(&body);
            let text = doc.text.trim().to_string();
            if text.is_empty() { rejected += 1; continue; }
            let article = make_article(&page.final_url, &doc.title, candidate.published_at.or(doc.published_at), &text, &source.id);
            if !article_is_acceptable(&article, settings) { rejected += 1; continue; }
            source.last_success = Some(article.fetched_at);
            articles.push(article);
        }
        if source.last_success.is_none() && articles.is_empty() && pages_scanned > 0 { errors.push("source produced no accepted articles".into()); }

        ScanResult { source, pages_scanned, rejected, articles, errors }
    }
}

pub struct SourceRegistry { path: PathBuf, sources: Vec<SourceRecord> }
impl SourceRegistry {
    pub fn load(root: &Path) -> Result<Self, String> {
        let dir = root.join("web_learning");
        fs::create_dir_all(&dir).map_err(|e| format!("create web directory: {e}"))?;
        let path = dir.join("sources.json");
        if !path.exists() {
            let sources: Vec<SourceRecord> = serde_json::from_str(include_str!("default_sources.json"))
                .map_err(|e| format!("default source parse: {e}"))?;
            let value = Self { path, sources };
            value.save()?;
            return Ok(value);
        }
        let text = fs::read_to_string(&path).map_err(|e| format!("read source registry: {e}"))?;
        let sources = serde_json::from_str(&text).map_err(|e| format!("parse source registry: {e}"))?;
        Ok(Self { path, sources })
    }
    pub fn save(&self) -> Result<(), String> {
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, serde_json::to_vec_pretty(&self.sources).map_err(|e| e.to_string())?)
            .map_err(|e| format!("write source registry: {e}"))?;
        fs::rename(tmp, &self.path).map_err(|e| format!("replace source registry: {e}"))
    }
    pub fn list(&self) -> &[SourceRecord] { &self.sources }
    pub fn add_url(&mut self, url: &str, language: Option<String>, topic: Option<String>, country: Option<String>) -> Result<String, String> {
        let canonical = canonicalize_url(url);
        if !is_http_url(&canonical) { return Err("source URL must be http or https".into()); }
        if let Some(existing) = self.sources.iter().find(|v| canonicalize_url(&v.url) == canonical) { return Ok(existing.id.clone()); }
        let domain = Url::parse(&canonical).ok().and_then(|v| v.host_str().map(str::to_string)).ok_or("source URL has no domain")?;
        let id = format!("src-{:016x}", fnv1a64(canonical.as_bytes()));
        self.sources.push(SourceRecord { id: id.clone(), url: canonical, domain, feed_url: None, language, topic, country, priority: SourcePriority::Normal, enabled: true, last_scan: None, last_success: None, error_count: 0 });
        self.save()?;
        Ok(id)
    }
    fn update(&mut self, source: SourceRecord) { if let Some(item) = self.sources.iter_mut().find(|v| v.id == source.id) { *item = source; } }
    pub fn discover_matching(&self, settings: &WebSettings) -> Vec<SourceRecord> {
        let all = settings.languages.iter().any(|v| v.eq_ignore_ascii_case("All"));
        self.sources.iter().filter(|source| {
            source.enabled
                && (all || source.language.as_deref().map(|v| settings.languages.iter().any(|l| l.eq_ignore_ascii_case(v))).unwrap_or(true))
                && settings.topic.as_deref().map(|topic| source.topic.as_deref().map(|v| v.eq_ignore_ascii_case(topic)).unwrap_or(true)).unwrap_or(true)
                && settings.country.as_deref().map(|country| source.country.as_deref().map(|v| v.eq_ignore_ascii_case(country)).unwrap_or(true)).unwrap_or(true)
        }).cloned().collect()
    }
}

#[derive(Clone, Debug)]
struct CacheEntry {
    etag: Option<String>,
    last_modified: Option<String>,
}

pub struct CorpusStore { root: PathBuf, db_path: PathBuf, corpus_path: PathBuf }
impl CorpusStore {
    pub fn open(root: &Path) -> Result<Self, String> {
        let root = root.to_path_buf();
        fs::create_dir_all(root.join("web_learning")).map_err(|e| e.to_string())?;
        fs::create_dir_all(root.join("web_corpus")).map_err(|e| e.to_string())?;
        let value = Self { db_path: root.join("web_learning/web.sqlite"), corpus_path: root.join("web_corpus/web.aicorpus"), root };
        value.init_db()?;
        Ok(value)
    }
    fn connection(&self) -> Result<Connection, String> { Connection::open(&self.db_path).map_err(|e| format!("open web sqlite: {e}")) }
    fn init_db(&self) -> Result<(), String> {
        self.connection()?.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS articles(
               id TEXT PRIMARY KEY,
               url TEXT NOT NULL,
               canonical_url TEXT NOT NULL UNIQUE,
               domain TEXT NOT NULL,
               title TEXT NOT NULL,
               published_at TEXT,
               fetched_at INTEGER NOT NULL,
               language TEXT NOT NULL,
               text TEXT NOT NULL,
               content_hash INTEGER NOT NULL UNIQUE,
               source_id TEXT NOT NULL,
               word_count INTEGER NOT NULL,
               token_count INTEGER NOT NULL,
               quality_score REAL NOT NULL,
               simhash INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS training_queue(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               article_id TEXT NOT NULL REFERENCES articles(id) ON DELETE CASCADE,
               state TEXT NOT NULL,
               token_count INTEGER NOT NULL,
               created_at INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_queue_state ON training_queue(state);
             CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
             CREATE TABLE IF NOT EXISTS page_cache(
               canonical_url TEXT PRIMARY KEY,
               etag TEXT,
               last_modified TEXT,
               content_hash INTEGER,
               fetched_at INTEGER NOT NULL
             );"
        ).map_err(|e| format!("initialize web sqlite: {e}"))
    }
    fn cache_entry(&self, url: &str) -> Result<Option<CacheEntry>, String> {
        self.connection()?.query_row(
            "SELECT etag,last_modified FROM page_cache WHERE canonical_url=?1",
            params![canonicalize_url(url)],
            |row| Ok(CacheEntry { etag: row.get(0)?, last_modified: row.get(1)? }),
        ).optional().map_err(|e| format!("query page cache: {e}"))
    }
    fn record_page_cache(&self, url: &str, etag: Option<&str>, last_modified: Option<&str>, content_hash: u64) -> Result<(), String> {
        self.connection()?.execute(
            "INSERT INTO page_cache(canonical_url,etag,last_modified,content_hash,fetched_at) VALUES(?1,?2,?3,?4,?5)
             ON CONFLICT(canonical_url) DO UPDATE SET etag=excluded.etag,last_modified=excluded.last_modified,content_hash=excluded.content_hash,fetched_at=excluded.fetched_at",
            params![canonicalize_url(url), etag, last_modified, content_hash as i64, now_ms() as i64]
        ).map_err(|e| format!("update page cache: {e}"))?;
        Ok(())
    }
    fn is_duplicate(&self, article: &ArticleRecord) -> Result<bool, String> {
        let conn = self.connection()?;
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM articles WHERE canonical_url=?1 OR content_hash=?2)",
            params![canonicalize_url(&article.canonical_url), article.content_hash as i64],
            |row| row.get(0)
        ).map_err(|e| format!("duplicate query: {e}"))?;
        if exists != 0 { return Ok(true); }
        let fingerprint = simhash(&article.text);
        let mut stmt = conn.prepare("SELECT simhash,word_count FROM articles ORDER BY fetched_at DESC LIMIT 1024")
            .map_err(|e| format!("prepare duplicate scan: {e}"))?;
        let rows = stmt.query_map([], |row| Ok((row.get::<_, i64>(0)? as u64, row.get::<_, i64>(1)? as usize)))
            .map_err(|e| format!("query duplicate scan: {e}"))?;
        for row in rows {
            let (other, words) = row.map_err(|e| format!("read duplicate scan: {e}"))?;
            if words.abs_diff(article.word_count) <= 24 && (fingerprint ^ other).count_ones() <= 6 { return Ok(true); }
        }
        Ok(false)
    }
    fn insert_article(&self, article: &ArticleRecord, token_ids: &[u32]) -> Result<bool, String> {
        let mut conn = self.connection()?;
        let tx = conn.transaction().map_err(|e| format!("article transaction: {e}"))?;
        let inserted = tx.execute(
            "INSERT OR IGNORE INTO articles(id,url,canonical_url,domain,title,published_at,fetched_at,language,text,content_hash,source_id,word_count,token_count,quality_score,simhash)
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15)",
            params![article.id,article.url,article.canonical_url,article.domain,article.title,article.published_at,article.fetched_at as i64,article.language,article.text,article.content_hash as i64,article.source_id,article.word_count as i64,token_ids.len() as i64,article.quality_score as f64,simhash(&article.text) as i64]
        ).map_err(|e| format!("insert article: {e}"))?;
        if inserted == 0 { return Ok(false); }
        if !token_ids.is_empty() {
            tx.execute(
                "INSERT INTO training_queue(article_id,state,token_count,created_at) VALUES(?1,'queued',?2,?3)",
                params![article.id, token_ids.len() as i64, now_ms() as i64]
            ).map_err(|e| format!("insert queue: {e}"))?;
        }
        tx.commit().map_err(|e| format!("commit article: {e}"))?;
        append_corpus(&self.corpus_path, article, token_ids)?;
        Ok(true)
    }
    pub fn stats(&self) -> Result<WebStats, String> {
        let conn = self.connection()?;
        let articles: i64 = conn.query_row("SELECT COUNT(*) FROM articles", [], |row| row.get(0)).map_err(|e| e.to_string())?;
        let queued: i64 = conn.query_row("SELECT COALESCE(SUM(token_count),0) FROM training_queue WHERE state='queued'", [], |row| row.get(0)).map_err(|e| e.to_string())?;
        let items: i64 = conn.query_row("SELECT COUNT(*) FROM training_queue WHERE state='queued'", [], |row| row.get(0)).map_err(|e| e.to_string())?;
        Ok(WebStats { articles_collected: articles.max(0) as u64, tokens_queued: queued.max(0) as u64, training_queue: items.max(0) as u64, last_update: Some(now_ms()), ..Default::default() })
    }
    pub fn mark_queue_state(&self, state: &str, limit: usize) -> Result<usize, String> {
        self.connection()?.execute(
            "UPDATE training_queue SET state=?1 WHERE id IN(
                SELECT id FROM training_queue WHERE state='queued' ORDER BY id LIMIT ?2
            )",
            params![state, limit as i64],
        ).map_err(|e| format!("update training queue: {e}"))
    }

    pub fn recover_inflight(&self) -> Result<usize, String> {
        self.connection()?.execute(
            "UPDATE training_queue SET state='queued' WHERE state='inflight'",
            []
        ).map_err(|e| format!("recover training queue: {e}"))
    }

    pub fn export_training_snapshot(&self, output: &Path, replay_percent: u8) -> Result<u64, String> {
        let conn = self.connection()?;
        let fresh = 100usize.saturating_sub(replay_percent as usize);
        let mut texts = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT a.text FROM training_queue q JOIN articles a ON a.id=q.article_id WHERE q.state='queued' ORDER BY q.created_at DESC LIMIT 4096").map_err(|e| e.to_string())?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0)).map_err(|e| e.to_string())?;
            for row in rows { texts.push(row.map_err(|e| e.to_string())?); }
        }
        let mut replay = Vec::new();
        {
            let mut stmt = conn.prepare("SELECT text FROM articles ORDER BY fetched_at ASC LIMIT 1024").map_err(|e| e.to_string())?;
            let rows = stmt.query_map([], |row| row.get::<_, String>(0)).map_err(|e| e.to_string())?;
            for row in rows { replay.push(row.map_err(|e| e.to_string())?); }
        }
        fs::create_dir_all(output.parent().unwrap_or(&self.root)).map_err(|e| e.to_string())?;
        let mut file = File::create(output).map_err(|e| e.to_string())?;
        let mut lines = 0u64;
        for (index,text) in texts.iter().enumerate() {
            if index % 100 < fresh { writeln!(file,"{text}").map_err(|e| e.to_string())?; lines += 1; }
        }
        for (index,text) in replay.iter().enumerate() {
            if index % 100 < replay_percent as usize { writeln!(file,"{text}").map_err(|e| e.to_string())?; lines += 1; }
        }
        Ok(lines)
    }
    pub fn mark_inflight_complete(&self, state: &str, limit: usize) -> Result<usize, String> {
        self.connection()?
            .execute(
                "UPDATE training_queue SET state=?1
                 WHERE id IN (
                    SELECT id FROM training_queue
                    WHERE state='inflight'
                    ORDER BY id
                    LIMIT ?2
                 )",
                params![state, limit as i64],
            )
            .map_err(|e| format!("update inflight queue: {e}"))
    }

    pub fn enforce_retention(&self, max_articles: usize) -> Result<usize, String> {
        let mut conn = self.connection()?;
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM articles", [], |row| row.get(0)).map_err(|e| e.to_string())?;
        let excess = (count.max(0) as usize).saturating_sub(max_articles);
        if excess == 0 { return Ok(0); }
        let tx = conn.transaction().map_err(|e| e.to_string())?;
        tx.execute("DELETE FROM articles WHERE id IN(SELECT id FROM articles ORDER BY fetched_at ASC LIMIT ?1)", params![excess as i64]).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(excess)
    }
}

pub struct TrainingBridge;
impl TrainingBridge {
    pub fn prepare_snapshot(root: &Path, replay_ratio_percent: u8) -> Result<(PathBuf,u64),String> {
        let store = CorpusStore::open(root)?;
        let path = root.join("web_learning/training_snapshot.txt");
        let lines = store.export_training_snapshot(&path, replay_ratio_percent)?;
        if lines != 0 {
            store.mark_queue_state("inflight", 4096)?;
        }
        Ok((path, lines))
    }

    pub fn finalize_batch(root: &Path, success: bool) -> Result<usize, String> {
        let store = CorpusStore::open(root)?;
        store.mark_inflight_complete(if success { "trained" } else { "queued" }, 4096)
    }
}

pub enum WebCommand { Start, Pause, Resume, Stop, ScanNow }

pub struct WebLearner { commands: Sender<WebCommand>, pub events: Receiver<WebEvent>, join: Option<thread::JoinHandle<()>> }
impl WebLearner {
    pub fn spawn(root: impl AsRef<Path>, mut settings: WebSettings) -> Result<Self,String> {
        settings.normalize();
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root).map_err(|e| e.to_string())?;
        let (command_tx, command_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();
        let join = thread::Builder::new().name("ainet-web-scheduler".into()).spawn(move || {
            run_scheduler(root, settings, command_rx, event_tx);
        }).map_err(|e| format!("spawn web scheduler: {e}"))?;
        Ok(Self { commands: command_tx, events: event_rx, join: Some(join) })
    }
    pub fn send(&self, command: WebCommand) -> Result<(),String> { self.commands.send(command).map_err(|e| format!("web command: {e}")) }
    pub fn stop_and_join(&mut self) {
        let _ = self.send(WebCommand::Stop);
        if let Some(join) = self.join.take() { let _ = join.join(); }
    }
}
impl Drop for WebLearner { fn drop(&mut self) { self.stop_and_join(); } }

fn run_scheduler(root: PathBuf, settings: WebSettings, command_rx: Receiver<WebCommand>, event_tx: Sender<WebEvent>) {
    let mut registry = match SourceRegistry::load(&root) { Ok(v)=>v, Err(e)=>{let _=event_tx.send(WebEvent::Error{source_id:None,error:e});return;} };
    let store = match CorpusStore::open(&root) { Ok(v)=>v, Err(e)=>{let _=event_tx.send(WebEvent::Error{source_id:None,error:e});return;} };
    let _ = store.recover_inflight();
    let tokenizer = settings.tokenizer_path.as_deref().and_then(|p| crate::tokenizer::Tokenizer::load(p).ok());
    let limiter = DomainLimiter::new();
    let (scan_tx, scan_rx) = mpsc::channel::<ScanResult>();
    let mut pending = VecDeque::<SourceRecord>::new();
    let mut active: Vec<thread::JoinHandle<()>> = Vec::new();
    let mut stats = store.stats().unwrap_or_default();
    stats.sources = registry.list().len() as u64;
    stats.active_sources = registry.list().iter().filter(|v| v.enabled).count() as u64;
    let _ = event_tx.send(WebEvent::Stats(stats.clone()));
    let _ = event_tx.send(WebEvent::Sources(registry.list().to_vec()));

    let mut paused = true;
    let mut next_scan = Instant::now();

    loop {
        while let Ok(command) = command_rx.try_recv() {
            match command {
                WebCommand::Start | WebCommand::Resume => {
                    paused = false;
                    let _ = event_tx.send(WebEvent::Status(WebStatus::Running));
                    next_scan = Instant::now();
                }
                WebCommand::Pause => {
                    paused = true;
                    let _ = event_tx.send(WebEvent::Status(WebStatus::Paused));
                }
                WebCommand::ScanNow => { next_scan = Instant::now(); }
                WebCommand::Stop => {
                    let _ = event_tx.send(WebEvent::Status(WebStatus::Stopping));
                    for handle in active.drain(..) { let _ = handle.join(); }
                    let _ = event_tx.send(WebEvent::Status(WebStatus::Stopped));
                    return;
                }
            }
        }

        let mut finished = Vec::new();
        for (idx, handle) in active.iter().enumerate() {
            if handle.is_finished() { finished.push(idx); }
        }
        for idx in finished.into_iter().rev() {
            let handle = active.swap_remove(idx);
            let _ = handle.join();
        }

        while let Ok(result) = scan_rx.try_recv() {
            stats.pages_scanned += result.pages_scanned;
            stats.pages_rejected += result.rejected;
            stats.current_source = Some(result.source.id.clone());
            stats.last_update = Some(now_ms());
            registry.update(result.source.clone());
            if let Err(error) = registry.save() { let _=event_tx.send(WebEvent::Error{source_id:None,error}); }
            if !result.errors.is_empty() {
                for error in result.errors {
                    let _ = event_tx.send(WebEvent::Error { source_id: Some(result.source.id.clone()), error: error.clone() });
                    if error.to_ascii_lowercase().contains("timed out") || error.to_ascii_lowercase().contains("connection") || error.to_ascii_lowercase().contains("dns") {
                        let _ = event_tx.send(WebEvent::Offline(error));
                    }
                }
            }
            stats.current_url = result.articles.last().map(|article| article.canonical_url.clone());
            for article in result.articles {
                if store.is_duplicate(&article).unwrap_or(true) {
                    stats.duplicates_skipped += 1;
                    continue;
                }
                let token_ids = tokenizer.as_ref().map(|t| t.encode(&article.text)).unwrap_or_default();
                let mut article = article;
                article.token_count = token_ids.len();
                match store.insert_article(&article, &token_ids) {
                    Ok(true) => {
                        stats.pages_accepted += 1;
                        stats.articles_collected += 1;
                        stats.tokens_queued += token_ids.len() as u64;
                        let _ = event_tx.send(WebEvent::ArticleAccepted(article));
                    }
                    Ok(false) => { stats.duplicates_skipped += 1; }
                    Err(error) => { let _=event_tx.send(WebEvent::Error{source_id:None,error}); }
                }
            }
            let persistent = store.stats().unwrap_or_default();
            stats.training_queue = persistent.training_queue;
            stats.last_update = Some(now_ms());
            let _ = store.enforce_retention(settings.retention_max_articles);
            let _ = event_tx.send(WebEvent::Stats(stats.clone()));
            let _ = event_tx.send(WebEvent::Sources(registry.list().to_vec()));
        }

        if !paused && pending.is_empty() && active.is_empty() && Instant::now() >= next_scan {
            let mut sources = registry.discover_matching(&settings);
            sources.sort_by_priority();
            pending = sources.into_iter().collect::<VecDeque<_>>();
            next_scan = Instant::now() + Duration::from_secs(settings.scan_interval_secs);
        }

        if !paused {
            let worker_limit = settings.maximum_concurrent_domains.min(settings.maximum_concurrent_requests).clamp(1,4);
            while active.len() < worker_limit && !pending.is_empty() {
                let source = pending.pop_front().unwrap_or_else(|| unreachable!());
                let settings_clone = settings.clone();
                let limiter_clone = limiter.clone();
                let tx = scan_tx.clone();
                let root_clone = root.clone();
                let handle = thread::Builder::new().name(format!("ainet-crawler-{}", source.id)).spawn(move || {
                    let result = WebDiscovery::scan_source(source, &settings_clone, limiter_clone, &root_clone);
                    let _ = tx.send(result);
                });
                if let Ok(handle) = handle { active.push(handle); }
            }
        }

        thread::sleep(Duration::from_millis(100));
    }
}

trait PrioritySort { fn sort_by_priority(&mut self); }
impl PrioritySort for Vec<SourceRecord> {
    fn sort_by_priority(&mut self) {
        self.sort_by_key(|source| match source.priority { SourcePriority::High=>0, SourcePriority::Normal=>1, SourcePriority::Low=>2 });
    }
}

fn fetch_cached(fetcher: &WebFetcher, store: &CorpusStore, url: &str) -> Result<Option<FetchedPage>,String> {
    let cache = store.cache_entry(url)?;
    let page = fetcher.fetch(url, cache.as_ref().and_then(|v| v.etag.as_deref()), cache.as_ref().and_then(|v| v.last_modified.as_deref()))?;
    if let Some(page) = &page {
        store.record_page_cache(url, page.etag.as_deref(), page.last_modified.as_deref(), fnv1a64(&page.bytes))?;
    }
    Ok(page)
}

fn article_is_acceptable(article: &ArticleRecord, settings: &WebSettings) -> bool {
    QualityFilter::accept(&article.text)
        && language_allowed(&article.language, &settings.languages)
        && topic_allowed(&article.text, settings.topic.as_deref())
        && freshness_allowed(article, settings.freshness_hours)
}

fn language_allowed(language: &str, languages: &[String]) -> bool {
    languages.iter().any(|v| v.eq_ignore_ascii_case("All") || v.eq_ignore_ascii_case(language))
}
fn language_source_allowed(source: &SourceRecord, languages: &[String]) -> bool {
    source.language.as_deref().map(|value| languages.iter().any(|v| v.eq_ignore_ascii_case("All") || v.eq_ignore_ascii_case(value))).unwrap_or(true)
}
fn topic_allowed(text: &str, topic: Option<&str>) -> bool {
    topic.map(|want| TopicClassifier::classify(text).eq_ignore_ascii_case(want)).unwrap_or(true)
}
fn freshness_allowed(article: &ArticleRecord, hours: u64) -> bool {
    if hours == 0 { return true; }
    let Some(date) = article.published_at.as_deref() else { return true };
    let date = date.get(0..10).unwrap_or(date);
    let Ok(y) = date.get(0..4).unwrap_or("").parse::<i32>() else { return true };
    let Ok(m) = date.get(5..7).unwrap_or("").parse::<u32>() else { return true };
    let Ok(d) = date.get(8..10).unwrap_or("").parse::<u32>() else { return true };
    let ordinal = |yy:i32,mm:u32,dd:u32| -> i64 {
        let mut total = 0i64;
        for year in 1970..yy { total += if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 }; }
        let month_days = [31i64,28,31,30,31,30,31,31,30,31,30,31];
        for month in 1..mm { total += month_days[(month-1) as usize] + if month==2 && yy%4==0 && (yy%100!=0||yy%400==0) {1} else {0}; }
        total + dd as i64 - 1
    };
    let article_day = ordinal(y,m,d);
    let now_day = now_ms() as i64 / 86_400_000;
    now_day.saturating_sub(article_day) <= hours.div_ceil(24) as i64
}

fn make_article(url:&str,title:&str,published_at:Option<String>,text:&str,source_id:&str)->ArticleRecord {
    let canonical_url = canonicalize_url(url);
    let cleaned = collapse_text(text);
    let content_hash = fnv1a64(cleaned.as_bytes());
    let fetched_at = now_ms();
    ArticleRecord {
        id: format!("article-{:016x}", fnv1a64(format!("{canonical_url}:{content_hash}").as_bytes())),
        url: url.into(),
        canonical_url: canonical_url.clone(),
        domain: Url::parse(&canonical_url).ok().and_then(|v| v.host_str().map(str::to_string)).unwrap_or_default(),
        title: title.trim().to_string(),
        published_at,
        fetched_at,
        language: LanguageDetector::detect(&cleaned),
        word_count: cleaned.split_whitespace().count(),
        token_count: 0,
        quality_score: QualityFilter::score(&cleaned),
        text: cleaned,
        content_hash,
        source_id: source_id.into(),
    }
}

fn robots_allowed(fetcher:&WebFetcher, limiter:&DomainLimiter, target:&str, delay_ms:u64)->Result<bool,String> {
    let url = Url::parse(target).map_err(|e| format!("robots URL: {e}"))?;
    let Some(host)=url.host_str() else { return Ok(false) };
    let mut robots=url.clone();
    robots.set_path("/robots.txt"); robots.set_query(None); robots.set_fragment(None);
    limiter.wait_turn(host, Duration::from_millis(delay_ms));
    match fetcher.fetch(robots.as_str(),None,None) {
        Ok(None) => Ok(true),
        Ok(Some(page)) => {
            let body = WebParser::decode(&page.bytes,&page.content_type)?;
            Ok(parse_robots(&body,target))
        }
        Err(error) if error.contains("HTTP status 404") || error.contains("HTTP status 410") => Ok(true),
        Err(error) => Err(error),
    }
}

fn parse_robots(body:&str,target:&str)->bool {
    let Some(url)=Url::parse(target).ok() else{return false};
    let path=format!("/{}",url.path().trim_start_matches('/'));
    let mut applies=false;
    let mut deny=Vec::<String>::new();
    for line in body.lines() {
        let line=line.split('#').next().unwrap_or_default().trim();
        let Some((key,value))=line.split_once(':') else{continue};
        match key.trim().to_ascii_lowercase().as_str() {
            "user-agent" => applies=value.trim()=="*" || value.trim().eq_ignore_ascii_case(USER_AGENT),
            "disallow" if applies => { let rule=value.trim(); if !rule.is_empty(){deny.push(rule.into());} },
            "allow" if applies => { let rule=value.trim(); if !rule.is_empty(){deny.retain(|v| !path.starts_with(v));} },
            _=>{}
        }
    }
    !deny.iter().any(|rule| path.starts_with(rule))
}

fn absolute_url(base:&str,candidate:&str)->String {
    Url::parse(candidate).ok().map(|v|v.to_string()).or_else(|| Url::parse(base).ok().and_then(|b| b.join(candidate).ok().map(|v|v.to_string()))).unwrap_or_else(||candidate.into())
}
fn canonicalize_url(input:&str)->String {
    let Ok(mut url)=Url::parse(input) else{return input.trim().into()};
    url.set_fragment(None);
    let remove=["utm_source","utm_medium","utm_campaign","utm_term","utm_content","gclid","fbclid","mc_cid","mc_eid"];
    let query:Vec<(String,String)>=url.query_pairs().filter(|(k,_)| !remove.iter().any(|v| v==k)).map(|(k,v)|(k.into_owned(),v.into_owned())).collect();
    let q=query.iter().map(|(k,v)|format!("{k}={v}")).collect::<Vec<_>>().join("&");
    url.set_query(if q.is_empty(){None}else{Some(&q)});
    url.to_string()
}
fn is_http_url(url:&str)->bool { Url::parse(url).map(|v|matches!(v.scheme(),"http"|"https")&&v.host_str().is_some()).unwrap_or(false) }
fn strip_html(input:&str)->String { ContentCleaner::clean(input).text }
fn collapse_text(input:&str)->String { input.split_whitespace().collect::<Vec<_>>().join(" ").trim().into() }
fn append_text(dst:&mut String,raw:&str){let value=raw.split_whitespace().collect::<Vec<_>>().join(" ");if value.is_empty(){return;}if !dst.is_empty(){dst.push(' ');}dst.push_str(&value);}
fn repeated_word_ratio(text:&str)->f32{let words:Vec<&str>=text.split_whitespace().collect();if words.len()<100{return 0.0;}let mut counts=HashMap::<&str,usize>::new();for w in &words{*counts.entry(w).or_default()+=1;}let repeated=counts.values().filter(|v|**v>=4).sum::<usize>();repeated as f32/words.len() as f32}
fn parse_tag(raw:&str)->(String,bool,String){let raw=raw.trim();let closing=raw.starts_with('/');let body=raw.trim_start_matches('/').trim_end_matches('/').trim();let mut parts=body.splitn(2,char::is_whitespace);(parts.next().unwrap_or_default().into(),closing,parts.next().unwrap_or_default().into())}
fn attr_value<'a>(attrs:&'a str,name:&str)->Option<&'a str>{let needle=format!("{name}=");attrs.split_whitespace().find_map(|v|v.strip_prefix(&needle).map(|v|v.trim_matches(['"','\''])))}
fn meta_pair(attrs:&str)->Option<(String,String)>{let key=attr_value(attrs,"property").or_else(||attr_value(attrs,"name"))?.to_string();let val=attr_value(attrs,"content")?.to_string();Some((key,val))}
fn tag_blocks<'a>(body:&'a str,tag:&str)->Vec<&'a str>{let mut out=Vec::new();let open=format!("<{tag}");let close=format!("</{tag}>");let mut cursor=0;while let Some(rel)=body[cursor..].find(&open){let start=cursor+rel;let begin=body[start..].find('>').map(|v|start+v+1).unwrap_or(body.len());let Some(end_rel)=body[begin..].find(&close) else{break};let end=begin+end_rel;out.push(&body[begin..end]);cursor=end+close.len();}out}
fn tag_values(body:&str,tag:&str)->Vec<String>{tag_blocks(body,tag).into_iter().map(strip_html).collect()}
fn extract_tag_value(body:&str,tag:&str)->Option<String>{tag_values(body,tag).into_iter().next()}
fn extract_attr_value(body:&str,tag:&str,attr:&str)->Option<String>{let open=format!("<{tag}");let start=body.find(&open)?;let end=body[start..].find('>')?+start;attr_value(&body[start+open.len()..end],attr).map(str::to_string)}
fn walk_json(value:&Value,out:&mut Vec<DiscoveredItem>){match value{Value::Array(items)=>{for item in items{walk_json(item,out);}},Value::Object(map)=>{let url=["url","link","href"].iter().find_map(|k|map.get(*k).and_then(Value::as_str));if let Some(url)=url.filter(|v|is_http_url(v)){out.push(DiscoveredItem{url:url.into(),title:map.get("title").and_then(Value::as_str).map(str::to_string),published_at:map.get("published_at").and_then(Value::as_str).or_else(||map.get("published").and_then(Value::as_str)).or_else(||map.get("date").and_then(Value::as_str)).map(str::to_string),content_hint:["content","text","description","summary"].iter().find_map(|k|map.get(*k).and_then(Value::as_str)).map(str::to_string)});}for child in map.values(){walk_json(child,out);}},_=>{}}}
fn decode_entities(input:&str)->String{input.replace("&nbsp;"," ").replace("&amp;","&").replace("&lt;","<").replace("&gt;",">").replace("&quot;","\"").replace("&#39;","'").replace("&apos;","'").replace("&#x27;","'")}
fn simhash(text:&str)->u64{let mut counts=[0i32;64];for word in text.split_whitespace(){let hash=fnv1a64(word.as_bytes());for bit in 0..64{if (hash>>bit)&1==1{counts[bit]+=1}else{counts[bit]-=1}}}let mut result=0;for bit in 0..64{if counts[bit]>=0{result|=1u64<<bit;}}result}
fn fnv1a64(data:&[u8])->u64{let mut hash=0xcbf29ce484222325u64;for byte in data{hash^=*byte as u64;hash=hash.wrapping_mul(0x100000001b3);}hash}
fn now_ms()->u64{SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64}
fn append_corpus(path:&Path,article:&ArticleRecord,tokens:&[u32])->Result<(),String>{let mut file=OpenOptions::new().create(true).append(true).open(path).map_err(|e|format!("open corpus: {e}"))?;file.write_all(CORPUS_MAGIC).map_err(|e|e.to_string())?;file.write_all(&CORPUS_VERSION.to_le_bytes()).map_err(|e|e.to_string())?;write_string(&mut file,&article.id)?;write_string(&mut file,&article.language)?;file.write_all(&article.fetched_at.to_le_bytes()).map_err(|e|e.to_string())?;file.write_all(&(tokens.len() as u32).to_le_bytes()).map_err(|e|e.to_string())?;for token in tokens{file.write_all(&token.to_le_bytes()).map_err(|e|e.to_string())?;}let mut data=Vec::with_capacity(article.id.len()+tokens.len()*4);data.extend_from_slice(article.id.as_bytes());for token in tokens{data.extend_from_slice(&token.to_le_bytes());}file.write_all(&fnv1a64(&data).to_le_bytes()).map_err(|e|e.to_string())?;Ok(())}
fn write_string(file:&mut File,value:&str)->Result<(),String>{if value.len()>u16::MAX as usize{return Err("corpus string too long".into())}file.write_all(&(value.len() as u16).to_le_bytes()).map_err(|e|e.to_string())?;file.write_all(value.as_bytes()).map_err(|e|e.to_string())}
