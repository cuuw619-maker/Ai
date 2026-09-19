#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DatasetCursor {
    pub dataset_id: String,
    pub file_path: String,
    pub file_offset: u64,
    pub sample_index: u64,
    pub token_position: u64,
}
