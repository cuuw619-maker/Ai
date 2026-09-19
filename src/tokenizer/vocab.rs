pub const BYTE_VOCAB_SIZE: u32 = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpecialToken {
    pub id: u32,
    pub name: String,
    pub text: String,
}

pub fn default_special_tokens() -> Vec<SpecialToken> {
    let values = [
        ("pad", "<|pad|>"),
        ("bos", "<|bos|>"),
        ("eos", "<|eos|>"),
        ("user", "<|user|>"),
        ("assistant", "<|assistant|>"),
        ("system", "<|system|>"),
    ];
    values
        .iter()
        .enumerate()
        .map(|(i, (name, text))| SpecialToken {
            id: BYTE_VOCAB_SIZE + i as u32,
            name: (*name).into(),
            text: (*text).into(),
        })
        .collect()
}
