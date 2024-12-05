use std::{env, fmt, fs, str::FromStr};
use thiserror::Error;

use anyhow::Result;

#[derive(Debug, Clone)]
/// The source of the HF token.
pub enum TokenSource {
    Literal(String),
    EnvVar(String),
    Path(String),
    CacheToken,
    None,
}

impl FromStr for TokenSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        match parts[0] {
            "literal" => parts
                .get(1)
                .map(|&value| TokenSource::Literal(value.to_string()))
                .ok_or_else(|| "Expected a value for 'literal'".to_string()),
            "env" => Ok(TokenSource::EnvVar(
                parts
                    .get(1)
                    .unwrap_or(&"HUGGING_FACE_HUB_TOKEN")
                    .to_string(),
            )),
            "path" => parts
                .get(1)
                .map(|&value| TokenSource::Path(value.to_string()))
                .ok_or_else(|| "Expected a value for 'path'".to_string()),
            "cache" => Ok(TokenSource::CacheToken),
            "none" => Ok(TokenSource::None),
            _ => Err("Invalid token source format".to_string()),
        }
    }
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenSource::Literal(value) => write!(f, "literal:{}", value),
            TokenSource::EnvVar(value) => write!(f, "env:{}", value),
            TokenSource::Path(value) => write!(f, "path:{}", value),
            TokenSource::CacheToken => write!(f, "cache"),
            TokenSource::None => write!(f, "none"),
        }
    }
}

#[derive(Error, Debug)]
enum TokenRetrievalError {
    #[error("No home directory.")]
    HomeDirectoryMissing,
}

/// This reads a token from a specified source. If the token cannot be read, a warning is logged with `tracing`
/// and *no token is used*.
pub(crate) fn get_token(source: &TokenSource) -> Result<Option<String>> {
    fn skip_token(input: &str) -> Option<String> {
        println!("Could not load token at {input:?}, using no HF token.");
        None
    }

    let token = match source {
        TokenSource::Literal(data) => Some(data.clone()),
        TokenSource::EnvVar(envvar) => env::var(envvar).ok().or_else(|| skip_token(envvar)),
        TokenSource::Path(path) => fs::read_to_string(path).ok().or_else(|| skip_token(path)),
        TokenSource::CacheToken => {
            let home = format!(
                "{}/.cache/huggingface/token",
                dirs::home_dir()
                    .ok_or(TokenRetrievalError::HomeDirectoryMissing)?
                    .display()
            );

            fs::read_to_string(home.clone())
                .ok()
                .or_else(|| skip_token(&home))
        }
        TokenSource::None => None,
    };

    Ok(token.map(|s| s.trim().to_string()))
}
