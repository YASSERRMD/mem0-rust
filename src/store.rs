use crate::embedder::Embedder;
use crate::errors::MemoryError;
use crate::models::{MemoryRecord, ScoredMemory};

fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, MemoryError> {
    if a.len() != b.len() {
        return Err(MemoryError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (va, vb) in a.iter().zip(b.iter()) {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

pub trait VectorStore {
    fn insert(&mut self, record: MemoryRecord, embedding: Vec<f32>);
    fn delete(&mut self, id: &str) -> Result<(), MemoryError>;
    fn search(
        &self,
        query_embedding: &[f32],
        max_results: usize,
    ) -> Result<Vec<ScoredMemory>, MemoryError>;
}

#[derive(Default)]
pub struct InMemoryStore {
    entries: Vec<(MemoryRecord, Vec<f32>)>,
}

impl VectorStore for InMemoryStore {
    fn insert(&mut self, record: MemoryRecord, embedding: Vec<f32>) {
        self.entries.push((record, embedding));
    }

    fn delete(&mut self, id: &str) -> Result<(), MemoryError> {
        let len_before = self.entries.len();
        self.entries
            .retain(|(record, _)| record.id.to_string() != id);
        if self.entries.len() == len_before {
            return Err(MemoryError::NotFound(id.to_string()));
        }
        Ok(())
    }

    fn search(
        &self,
        query_embedding: &[f32],
        max_results: usize,
    ) -> Result<Vec<ScoredMemory>, MemoryError> {
        let mut scored: Vec<ScoredMemory> = self
            .entries
            .iter()
            .map(|(record, embedding)| {
                cosine_similarity(query_embedding, embedding).map(|score| ScoredMemory {
                    record: record.clone(),
                    score,
                })
            })
            .collect::<Result<_, _>>()?;

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(max_results);
        Ok(scored)
    }
}

pub fn embed_and_store(
    store: &mut impl VectorStore,
    embedder: &impl Embedder,
    record: MemoryRecord,
) -> MemoryRecord {
    let embedding = embedder.embed(&record.content);
    store.insert(record.clone(), embedding);
    record
}

pub fn search_store(
    store: &impl VectorStore,
    embedder: &impl Embedder,
    query: &str,
    max_results: usize,
) -> Result<Vec<ScoredMemory>, MemoryError> {
    let query_embedding = embedder.embed(query);
    store.search(&query_embedding, max_results)
}
