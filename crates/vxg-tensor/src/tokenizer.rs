//! Zero-dependency GPT-2 / Qwen Byte-Level BPE Tokenizer (`encode`, `encode_chat`, `decode`).

use std::collections::HashMap;

use crate::gguf_meta::{decode_string_list, encode_string_list, GgufVal};
use crate::{Error, Result};

pub struct BpeTokenizer {
    vocab: Vec<String>,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    special_tokens: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl BpeTokenizer {
    pub fn from_vocab_and_merges(
        vocab: Vec<String>,
        merges: &[String],
        token_types: Option<&[i32]>,
    ) -> Result<Self> {
        if vocab.is_empty() {
            return Err(Error::Format("empty BPE vocabulary".into()));
        }
        let mut token_to_id = HashMap::with_capacity(vocab.len() * 2);
        let mut special_tokens = Vec::new();
        for (i, tok) in vocab.iter().enumerate() {
            let id = i as u32;
            token_to_id.entry(tok.clone()).or_insert(id);
            let is_special = token_types
                .and_then(|tt| tt.get(i))
                .is_some_and(|&t| t == 3 || t == 4)
                || (tok.starts_with('<') && tok.ends_with('>') && tok.len() >= 3);
            if is_special && !tok.is_empty() {
                special_tokens.push((tok.clone(), id));
            }
        }
        // Longest special token first to avoid prefix shadowing
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut merge_ranks = HashMap::with_capacity(merges.len() * 2);
        for (rank, line) in merges.iter().enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                merge_ranks.insert((a.to_owned(), b.to_owned()), rank as u32);
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_maps();
        Ok(Self {
            vocab,
            token_to_id,
            merge_ranks,
            special_tokens,
            byte_to_char,
            char_to_byte,
        })
    }

    pub fn from_gguf_kv(kv: &HashMap<String, GgufVal>) -> Result<Self> {
        let vocab = match kv.get("tokenizer.ggml.tokens") {
            Some(GgufVal::ArrStr(v)) => v.clone(),
            _ => {
                return Err(Error::Format(
                    "missing tokenizer.ggml.tokens in GGUF".into(),
                ))
            }
        };
        let merges = match kv.get("tokenizer.ggml.merges") {
            Some(GgufVal::ArrStr(m)) => m.as_slice(),
            _ => &[],
        };
        let ttypes = match kv.get("tokenizer.ggml.token_type") {
            Some(GgufVal::ArrI32(t)) => Some(t.as_slice()),
            _ => None,
        };
        Self::from_vocab_and_merges(vocab, merges, ttypes)
    }

    pub fn to_vxt_meta_entries(kv: &HashMap<String, GgufVal>) -> Result<Vec<(String, Vec<u8>)>> {
        let vocab = match kv.get("tokenizer.ggml.tokens") {
            Some(GgufVal::ArrStr(v)) => v,
            _ => return Err(Error::Format("missing tokenizer.ggml.tokens".into())),
        };
        let empty = Vec::new();
        let merges = match kv.get("tokenizer.ggml.merges") {
            Some(GgufVal::ArrStr(m)) => m,
            _ => &empty,
        };
        let mut out = vec![
            ("tokenizer.tokens".to_owned(), encode_string_list(vocab)),
            ("tokenizer.merges".to_owned(), encode_string_list(merges)),
        ];
        if let Some(GgufVal::ArrI32(tt)) = kv.get("tokenizer.ggml.token_type") {
            let mut b = Vec::with_capacity(tt.len() * 4);
            for &v in tt {
                b.extend_from_slice(&v.to_le_bytes());
            }
            out.push(("tokenizer.types".to_owned(), b));
        }
        Ok(out)
    }

    pub fn from_vxt_meta(meta: &[(String, Vec<u8>)]) -> Result<Self> {
        let mut vocab = None;
        let mut merges = Vec::new();
        let mut ttypes: Option<Vec<i32>> = None;
        for (k, v) in meta {
            match k.as_str() {
                "tokenizer.tokens" => vocab = Some(decode_string_list(v)?),
                "tokenizer.merges" => merges = decode_string_list(v)?,
                "tokenizer.types" => {
                    ttypes = Some(
                        v.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                    );
                }
                _ => {}
            }
        }
        let vocab = vocab.ok_or_else(|| Error::Format("missing tokenizer.tokens in VXT".into()))?;
        Self::from_vocab_and_merges(vocab, &merges, ttypes.as_deref())
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn token_id(&self, tok: &str) -> Option<u32> {
        self.token_to_id.get(tok).copied()
    }

    /// Formats a user prompt using Qwen ChatML (`enable_thinking=false` by default for crisp output).
    pub fn encode_chat(&self, user_prompt: &str, arch: &str, enable_thinking: bool) -> Vec<u32> {
        if self.token_id("<|im_start|>").is_some() && self.token_id("<|im_end|>").is_some() {
            let suffix = if matches!(arch, "qwen3" | "qwen35") {
                if enable_thinking {
                    "<|im_start|>assistant\n<think>\n"
                } else {
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                }
            } else {
                "<|im_start|>assistant\n"
            };
            let formatted = format!("<|im_start|>user\n{user_prompt}<|im_end|>\n{suffix}");
            self.encode(&formatted)
        } else {
            self.encode(user_prompt)
        }
    }

    /// Encodes UTF-8 text (including embedded special tokens like `<|im_start|>`) into token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut cursor = 0usize;
        let bytes = text.as_bytes();

        while cursor < bytes.len() {
            // Check if any special token starts at or after cursor
            let rem = &text[cursor..];
            let mut earliest: Option<(usize, usize, u32)> = None;
            if rem.contains('<') {
                for &(ref sp, id) in &self.special_tokens {
                    if let Some(pos) = rem.find(sp.as_str()) {
                        if earliest.is_none_or(|(best_pos, _, _)| pos < best_pos) {
                            earliest = Some((pos, sp.len(), id));
                            if pos == 0 {
                                break;
                            }
                        }
                    }
                }
            }

            if let Some((pos, len, sp_id)) = earliest {
                if pos > 0 {
                    self.encode_ordinary(&rem[..pos], &mut ids);
                }
                ids.push(sp_id);
                cursor += pos + len;
            } else {
                self.encode_ordinary(rem, &mut ids);
                break;
            }
        }
        ids
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for piece in split_qwen_pretokens(text) {
            if piece.is_empty() {
                continue;
            }
            let mut parts: Vec<String> = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_to_char[b as usize].to_string())
                .collect();

            while parts.len() >= 2 {
                let mut best_idx = usize::MAX;
                let mut best_rank = u32::MAX;
                for i in 0..(parts.len() - 1) {
                    if let Some(&rank) = self
                        .merge_ranks
                        .get(&(parts[i].clone(), parts[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }
                if best_idx == usize::MAX {
                    break;
                }
                let merged = format!("{}{}", parts[best_idx], parts[best_idx + 1]);
                parts[best_idx] = merged;
                parts.remove(best_idx + 1);
            }

            for p in parts {
                if let Some(&id) = self.token_to_id.get(&p) {
                    out.push(id);
                }
            }
        }
    }

    /// Decodes a sequence of token IDs back into valid UTF-8 text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut raw_bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let Some(tok) = self.vocab.get(id as usize) else {
                continue;
            };
            if self.special_tokens.iter().any(|(_, sid)| *sid == id) {
                raw_bytes.extend_from_slice(tok.as_bytes());
                continue;
            }
            for ch in tok.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    raw_bytes.push(b);
                } else {
                    let mut u8b = [0u8; 4];
                    raw_bytes.extend_from_slice(ch.encode_utf8(&mut u8b).as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }
}

/// Qwen / GPT-4 style pre-tokenizer: groups `'s|'t|'re|'ve|'m|'ll|'d`, optional-space + letters,
/// single digits (`Qwen3/3.5` splits individual digits!), optional-space + punctuation, and newlines/spaces.
fn split_qwen_pretokens(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let end_byte = |idx: usize| -> usize {
        if idx < n {
            chars[idx].0
        } else {
            text.len()
        }
    };

    let mut i = 0usize;
    while i < n {
        let c = chars[i].1;
        // 1. Newlines (\r / \n)
        if c == '\r' || c == '\n' {
            let start = chars[i].0;
            while i < n && (chars[i].1 == '\r' || chars[i].1 == '\n') {
                i += 1;
            }
            out.push(&text[start..end_byte(i)]);
            continue;
        }
        // 2. Single ASCII/Unicode digit (`\p{N}` in Qwen3/3.5 pre-tokenizer is 1 digit per token!)
        if c.is_numeric() {
            let start = chars[i].0;
            i += 1;
            out.push(&text[start..end_byte(i)]);
            continue;
        }
        // 3. Optional single leading non-newline space followed by letters or punctuation
        if c == ' ' && i + 1 < n {
            let next = chars[i + 1].1;
            if next.is_alphabetic() {
                let start = chars[i].0;
                i += 2;
                while i < n && chars[i].1.is_alphabetic() {
                    i += 1;
                }
                out.push(&text[start..end_byte(i)]);
                continue;
            }
            if !next.is_whitespace() && !next.is_numeric() {
                let start = chars[i].0;
                i += 2;
                while i < n
                    && !chars[i].1.is_whitespace()
                    && !chars[i].1.is_alphabetic()
                    && !chars[i].1.is_numeric()
                {
                    i += 1;
                }
                out.push(&text[start..end_byte(i)]);
                continue;
            }
        }
        // 4. Pure whitespace run (excluding trailing space before word if next is letter)
        if c.is_whitespace() {
            let start = chars[i].0;
            while i < n && chars[i].1.is_whitespace() && chars[i].1 != '\r' && chars[i].1 != '\n' {
                if i > 0
                    && chars[i].0 > start
                    && chars[i].1 == ' '
                    && i + 1 < n
                    && !chars[i + 1].1.is_whitespace()
                {
                    break;
                }
                i += 1;
            }
            out.push(&text[start..end_byte(i)]);
            continue;
        }
        // 5. Letters
        if c.is_alphabetic() {
            let start = chars[i].0;
            i += 1;
            while i < n && chars[i].1.is_alphabetic() {
                i += 1;
            }
            out.push(&text[start..end_byte(i)]);
            continue;
        }
        // 6. Contractions ('s, 't, 'm, 'd, 'll, 've, 're) or punctuation
        let start = chars[i].0;
        if c == '\'' && i + 1 < n {
            let n1 = chars[i + 1].1.to_ascii_lowercase();
            if matches!(n1, 's' | 't' | 'm' | 'd') {
                i += 2;
                out.push(&text[start..end_byte(i)]);
                continue;
            }
            if i + 2 < n {
                let n2 = chars[i + 2].1.to_ascii_lowercase();
                if matches!((n1, n2), ('l', 'l') | ('v', 'e') | ('r', 'e')) {
                    i += 3;
                    out.push(&text[start..end_byte(i)]);
                    continue;
                }
            }
        }
        i += 1;
        while i < n
            && !chars[i].1.is_whitespace()
            && !chars[i].1.is_alphabetic()
            && !chars[i].1.is_numeric()
        {
            i += 1;
        }
        out.push(&text[start..end_byte(i)]);
    }
    out
}

fn build_gpt2_byte_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut b2c = ['\0'; 256];
    let mut c2b = HashMap::with_capacity(512);
    let mut extra = 0u32;
    for b in 0u16..=255 {
        let direct =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        let ch = if direct {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        b2c[b as usize] = ch;
        c2b.insert(ch, b as u8);
    }
    (b2c, c2b)
}
