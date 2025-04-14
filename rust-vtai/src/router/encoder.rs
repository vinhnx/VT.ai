use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use crate::utils::error::{Result, VTError};
use crate::router::constants::DEFAULT_EMBEDDING_MODEL;

/// Trait for text embedders
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a single text string
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple text strings
    async fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(texts.len());

        for text in texts {
            let embedding = self.embed_text(text).await?;
            results.push(embedding);
        }

        Ok(results)
    }

    /// Compute cosine similarity between two vectors
    fn cosine_similarity(&self, vec1: &[f32], vec2: &[f32]) -> f32 {
        if vec1.len() != vec2.len() || vec1.is_empty() {
            return 0.0;
        }

        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;

        for i in 0..vec1.len() {
            dot_product += vec1[i] as f64 * vec2[i] as f64;
            norm1 += vec1[i] as f64 * vec1[i] as f64;
            norm2 += vec2[i] as f64 * vec2[i] as f64;
        }

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        (dot_product / (norm1.sqrt() * norm2.sqrt())) as f32
    }
}

/// Simple mock embedder for development
pub struct MockEmbedder {
    dimension: usize,
}

impl MockEmbedder {
    /// Create a new mock embedder
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
        }
    }
}

#[async_trait]
impl Embedder for MockEmbedder {
    async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        // Generate a deterministic "embedding" based on the text
        // This is only for testing - not for production use
        let mut result = Vec::with_capacity(self.dimension);

        // Use a simple hash of the text to seed the values
        let hash = text.bytes().fold(0u64, |acc, b| acc.wrapping_add(b as u64));

        for i in 0..self.dimension {
            // Generate a value between -1.0 and 1.0 based on the hash and position
            let val = ((hash.wrapping_add(i as u64) % 1000) as f32 / 500.0) - 1.0;
            result.push(val);
        }

        // Normalize the vector
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut result {
            *val /= norm;
        }

        Ok(result)
    }
}

/// Factory for creating embedders
pub struct EmbedderFactory;

impl EmbedderFactory {
    /// Create an embedder based on the model name
    pub fn create(model_name: Option<&str>) -> Result<Arc<dyn Embedder>> {
        let model = model_name.unwrap_or(DEFAULT_EMBEDDING_MODEL);

        // For now, just return a mock embedder
        // In the future, we can add support for actual embedding models
        let embedder = MockEmbedder::new(384); // 384 is a common embedding dimension

        Ok(Arc::new(embedder))
    }
}