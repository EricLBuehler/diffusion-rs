use std::collections::HashMap;

use tokenizers::{models::bpe::BPE, ModelWrapper, Tokenizer};

use crate::{FileData, ModelSource};

pub fn load_bpe_tokenizer(
    vocab_file: &FileData,
    merges_file: &FileData,
    src: &ModelSource,
) -> anyhow::Result<Tokenizer> {
    let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_file.read_to_string(src)?)?;
    let merges: Vec<(String, String)> = merges_file
        .read_to_string(src)?
        .split('\n')
        .skip(1)
        .map(|x| x.split(' ').collect::<Vec<_>>())
        .filter(|x| x.len() == 2)
        .map(|x| (x[0].to_string(), x[1].to_string()))
        .collect();

    Ok(Tokenizer::new(ModelWrapper::BPE(BPE::new(vocab, merges))))
}
