use super::merge::Merge;
use super::tokenizer::Tokenizer;
use super::vocab::SpecialToken;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"AITOKv01";
const VERSION: u32 = 1;

pub fn encode_payload(tokenizer: &Tokenizer) -> Result<Vec<u8>, String> {
    let mut w = Writer::default();
    w.u32(tokenizer.vocab_size() as u32);
    w.u32(256);
    w.u32(tokenizer.special_tokens.len() as u32);
    for token in &tokenizer.special_tokens {
        w.u32(token.id);
        w.string(&token.name);
        w.string(&token.text);
    }
    w.u32(tokenizer.merges.len() as u32);
    for merge in &tokenizer.merges {
        w.u32(merge.left);
        w.u32(merge.right);
        w.u32(merge.new_id);
    }
    Ok(w.bytes)
}

pub fn save(tokenizer: &Tokenizer, path: &Path) -> Result<(), String> {
    let payload = encode_payload(tokenizer)?;
    let checksum = fnv1a64(&payload);
    let mut bytes = Vec::with_capacity(28 + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&payload);

    let temp = path.with_extension("aitok.tmp");
    {
        let mut file = File::create(&temp).map_err(|e| format!("create tokenizer temp: {e}"))?;
        file.write_all(&bytes).map_err(|e| format!("write tokenizer temp: {e}"))?;
        file.flush().map_err(|e| format!("flush tokenizer temp: {e}"))?;
        file.sync_all().map_err(|e| format!("sync tokenizer temp: {e}"))?;
    }
    fs::rename(&temp, path).map_err(|e| format!("install tokenizer: {e}"))?;
    Ok(())
}

pub fn load(path: &Path) -> Result<Tokenizer, String> {
    let bytes = fs::read(path).map_err(|e| format!("read tokenizer: {e}"))?;
    if bytes.len() < 28 || &bytes[..8] != MAGIC {
        return Err("invalid .aitok magic".into());
    }
    let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    if version != VERSION {
        return Err(format!("unsupported .aitok version {version}"));
    }
    let expected_checksum = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    let payload_len = u64::from_le_bytes(bytes[20..28].try_into().unwrap()) as usize;
    if bytes.len() != payload_len + 28 {
        return Err("invalid .aitok payload length".into());
    }
    let payload = &bytes[28..];
    if fnv1a64(payload) != expected_checksum {
        return Err("tokenizer checksum mismatch".into());
    }

    let mut r = Reader::new(payload);
    let vocab_size = r.u32()? as usize;
    let byte_vocab = r.u32()?;
    if byte_vocab != 256 {
        return Err("unsupported byte vocabulary size".into());
    }
    let special_count = r.u32()? as usize;
    let mut specials = Vec::with_capacity(special_count);
    for _ in 0..special_count {
        specials.push(SpecialToken {
            id: r.u32()?,
            name: r.string()?,
            text: r.string()?,
        });
    }
    let merge_count = r.u32()? as usize;
    let mut merges = Vec::with_capacity(merge_count);
    for _ in 0..merge_count {
        merges.push(Merge {
            left: r.u32()?,
            right: r.u32()?,
            new_id: r.u32()?,
        });
    }
    if vocab_size != 256 + special_count + merge_count {
        return Err("tokenizer vocabulary size metadata mismatch".into());
    }
    if !r.finished() {
        return Err("trailing bytes in .aitok".into());
    }
    Tokenizer::from_parts(specials, merges)
}

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
}
impl Writer {
    fn u32(&mut self, v: u32) { self.bytes.extend_from_slice(&v.to_le_bytes()); }
    fn string(&mut self, v: &str) {
        self.u32(v.len() as u32);
        self.bytes.extend_from_slice(v.as_bytes());
    }
}

struct Reader<'a> { data: &'a [u8], pos: usize }
impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() { return Err("truncated .aitok payload".into()); }
        let out = &self.data[self.pos..self.pos+n];
        self.pos += n;
        Ok(out)
    }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec()).map_err(|e| format!("invalid UTF-8 tokenizer string: {e}"))
    }
    fn finished(&self) -> bool { self.pos == self.data.len() }
}
fn fnv1a64(data: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in data { hash ^= *byte as u64; hash = hash.wrapping_mul(0x100000001b3); }
    hash
}
