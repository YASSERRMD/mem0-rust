//! Core Memory manager.

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};
use uuid::Uuid;
use chrono::Utc;

use crate::config::MemoryConfig;
use crate::embeddings::{create_embedder, Embedder};
use crate::errors::{LLMError, MemoryError};
use crate::history::HistoryManager;
use crate::llms::{create_llm, generate_json, GenerateOptions, LLM};
use crate::models::{
    AddOptions, AddResult, EventType, Filters, GetAllOptions, HistoryEntry, MemoryEvent,
    MemoryRecord, Message, Messages, Payload, ResetOptions, Role, ScoredMemory, SearchOptions,
    SearchResult,
};
use crate::vector_stores::{create_vector_store, VectorStore};
use crate::rerankers::{create_reranker, Reranker};

use super::prompts::{
    format_fact_extraction_input, format_memory_update_input, FACT_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
};

/// Main Memory interface
pub struct Memory {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    llm: Option<Arc<dyn LLM>>,
    history: Option<Arc<HistoryManager>>,
    reranker: Option<Arc<dyn Reranker>>,
    #[allow(dead_code)]
    config: MemoryConfig,
}

impl Memory {
    /// Create a new Memory instance
    pub async fn new(config: MemoryConfig) -> Result<Self, MemoryError> {
        let embedder = create_embedder(&config.embedder)?;
        let dimensions = embedder.dimensions();

        let vector_store =
            create_vector_store(&config.vector_store, &config.collection_name, dimensions).await?;

        let llm = if let Some(llm_config) = &config.llm {
            Some(create_llm(llm_config)?)
        } else {
            None
        };

        let history = if let Some(path) = &config.history_db_path {
            Some(Arc::new(HistoryManager::new(path)?))
        } else {
            None
        };

        let reranker = if let Some(reranker_config) = &config.reranker {
            Some(create_reranker(reranker_config)?)
        } else {
            None
        };

        info!(
            "Initialized Memory with {} embedder, {} dimensions",
            embedder.model_name(),
            dimensions
        );

        Ok(Self {
            embedder,
            vector_store,
            llm,
            history,
            reranker,
            config,
        })
    }

    /// Add memories from messages
    pub async fn add(
        &self,
        messages: impl Into<Messages>,
        options: AddOptions,
    ) -> Result<AddResult, MemoryError> {
        let messages = messages.into().into_messages();
        // Validate scoping
        if options.user_id.is_none() && options.agent_id.is_none() && options.run_id.is_none() {
            return Err(MemoryError::InvalidInput(
                "At least one of user_id, agent_id, or run_id is required".to_string(),
            ));
        }

        let results = if options.infer && self.llm.is_some() {
            // Use LLM for fact extraction
            self.add_with_inference(&messages, &options).await?
        } else {
            // Add messages directly without inference
            self.add_raw(&messages, &options).await?
        };

        Ok(AddResult { results })
    }

    /// Add messages directly without LLM inference
    async fn add_raw(
        &self,
        messages: &[Message],
        options: &AddOptions,
    ) -> Result<Vec<MemoryEvent>, MemoryError> {
        let mut results = Vec::new();

        for msg in messages {
            if msg.role == Role::System {
                continue;
            }

            let record = MemoryRecord::with_scoping(
                msg.content.clone(),
                options
                    .metadata
                    .as_ref()
                    .map(|m| serde_json::to_value(m).unwrap_or_default())
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                options.user_id.clone(),
                options.agent_id.clone(),
                options.run_id.clone(),
            );

            let embedding = self.embedder.embed(&record.content).await?;
            let payload = Payload::from(&record);

            self.vector_store
                .insert(&record.id.to_string(), embedding, payload)
                .await?;

            if let Some(history) = &self.history {
                let _ = history.add_history(
                    record.id,
                    None,
                    record.content.clone(),
                    EventType::Add,
                    record.created_at,
                    record.user_id.clone(),
                    record.agent_id.clone(),
                    record.run_id.clone(),
                );
            }

            results.push(MemoryEvent {
                id: record.id,
                memory: record.content,
                event: EventType::Add,
            });
        }

        Ok(results)
    }

    /// Add messages with LLM inference
    async fn add_with_inference(
        &self,
        messages: &[Message],
        options: &AddOptions,
    ) -> Result<Vec<MemoryEvent>, MemoryError> {
        let llm = self.llm.as_ref().ok_or(LLMError::NotConfigured)?;

        // Format messages for extraction
        let messages_text = messages
            .iter()
            .map(|m| format!("{:?}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        // Extract facts
        let extraction_messages = vec![
            Message::system(FACT_EXTRACTION_PROMPT),
            Message::user(format_fact_extraction_input(&messages_text)),
        ];

        #[derive(serde::Deserialize)]
        struct FactsResponse {
            facts: Vec<String>,
        }

        let facts: FactsResponse = generate_json(
            llm.as_ref(),
            &extraction_messages,
            GenerateOptions::default(),
        )
        .await?;

        if facts.facts.is_empty() {
            debug!("No facts extracted from messages");
            return Ok(Vec::new());
        }

        info!("Extracted {} facts", facts.facts.len());

        // Search for existing related memories
        let mut existing_memories: Vec<(String, String)> = Vec::new(); // (Index, Content)
        let mut memory_map: HashMap<String, String> = HashMap::new(); // Index -> RealID

        let search_filters = Filters {
            conditions: vec![],
            logic: crate::models::FilterLogic::And,
        };

        for fact in &facts.facts {
            let embedding = self.embedder.embed(fact).await?;

            let similar = self
                .vector_store
                .search(&embedding, 5, Some(&search_filters))
                .await?;

            for result in similar {
                // Check if we already have this memory in our list (dedupe by real ID)
                let real_id = result.id.clone();
                if !memory_map.values().any(|rid| rid == &real_id) {
                     let index = memory_map.len().to_string();
                     memory_map.insert(index.clone(), real_id);
                     existing_memories.push((index, result.payload.data));
                }
            }
        }

        // Determine memory actions
        let update_messages = vec![
            Message::system(MEMORY_UPDATE_PROMPT),
            Message::user(format_memory_update_input(&existing_memories, &facts.facts)),
        ];

        #[derive(serde::Deserialize)]
        struct MemoryAction {
            event: String,
            text: Option<String>,
            id: Option<String>,
        }

        #[derive(serde::Deserialize)]
        struct MemoryActionsResponse {
            memory: Vec<MemoryAction>,
        }

        let actions: MemoryActionsResponse = generate_json(
            llm.as_ref(),
            &update_messages,
            GenerateOptions::default(),
        )
        .await?;

        let mut results = Vec::new();

        for action in actions.memory {
            match action.event.to_uppercase().as_str() {
                "ADD" => {
                    if let Some(text) = action.text {
                        let record = MemoryRecord::with_scoping(
                            &text,
                            options
                                .metadata
                                .as_ref()
                                .map(|m| serde_json::to_value(m).unwrap_or_default())
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                            options.user_id.clone(),
                            options.agent_id.clone(),
                            options.run_id.clone(),
                        );

                        let embedding = self.embedder.embed(&text).await?;
                        let payload = Payload::from(&record);

                        self.vector_store
                            .insert(&record.id.to_string(), embedding, payload)
                            .await?;

                        if let Some(history) = &self.history {
                            let _ = history.add_history(
                                record.id,
                                None,
                                record.content.clone(),
                                EventType::Add,
                                record.created_at,
                                record.user_id.clone(),
                                record.agent_id.clone(),
                                record.run_id.clone(),
                            );
                        }

                        results.push(MemoryEvent {
                            id: record.id,
                            memory: text,
                            event: EventType::Add,
                        });
                    }
                }
                "UPDATE" => {
                    if let (Some(index_id), Some(text)) = (action.id, action.text) {
                        if let Some(real_id) = memory_map.get(&index_id) {
                            debug!("Updating memory {} (index {}) with: {}", real_id, index_id, text);
                            
                            // Perform update
                            match self.update(real_id, &text).await {
                                Ok(record) => {
                                    results.push(MemoryEvent {
                                        id: record.id,
                                        memory: text,
                                        event: EventType::Update,
                                    });
                                },
                                Err(e) => {
                                    warn!("Failed to update memory {}: {}", real_id, e);
                                }
                            }
                        } else {
                            warn!("LLM tried to update unknown memory index: {}", index_id);
                        }
                    }
                }
                "DELETE" => {
                    if let Some(index_id) = action.id {
                        if let Some(real_id) = memory_map.get(&index_id) {
                            debug!("Deleting memory {} (index {})", real_id, index_id);
                            
                            // Perform delete
                            match self.delete(real_id).await {
                                Ok(_) => {
                                     // ID is needed for event, but delete returns void.
                                     // We can use Uuid::parse_str(real_id)
                                     if let Ok(uuid) = Uuid::parse_str(real_id) {
                                         results.push(MemoryEvent {
                                            id: uuid,
                                            memory: String::new(), // Deleted
                                            event: EventType::Delete,
                                        });
                                     }
                                },
                                Err(e) => {
                                    warn!("Failed to delete memory {}: {}", real_id, e);
                                }
                            }
                        } else {
                            warn!("LLM tried to delete unknown memory index: {}", index_id);
                        }
                    }
                }
                "NOOP" => {
                    debug!("No action needed");
                }
                _ => {
                    warn!("Unknown memory action: {}", action.event);
                }
            }
        }

        Ok(results)
    }

    /// Search for memories
    pub async fn search(
        &self,
        query: &str,
        options: SearchOptions,
    ) -> Result<SearchResult, MemoryError> {
        let embedding = self.embedder.embed(query).await?;
        let limit = options.limit.unwrap_or(10);
        let threshold = options.threshold.unwrap_or(0.0);

        // Fetch more candidates if reranking is enabled
        let search_limit = if options.rerank { limit * 10 } else { limit * 2 };

        let results = self
            .vector_store
            .search(&embedding, search_limit, options.filters.as_ref())
            .await?;

        let mut scored: Vec<ScoredMemory> = results
            .into_iter()
            .map(|r| r.to_scored_memory())
            .collect();

        // Apply scoping filters
        scored.retain(|m| {
            if let Some(ref user_id) = options.user_id {
                if m.record.user_id.as_ref() != Some(user_id) {
                    return false;
                }
            }
            if let Some(ref agent_id) = options.agent_id {
                if m.record.agent_id.as_ref() != Some(agent_id) {
                    return false;
                }
            }
            if let Some(ref run_id) = options.run_id {
                if m.record.run_id.as_ref() != Some(run_id) {
                    return false;
                }
            }
            true
        });

        // Filter by threshold before reranking (optional, but saves rerank quota)
        scored.retain(|m| m.score >= threshold);

        // Reranking
        if options.rerank {
            if let Some(reranker) = &self.reranker {
                scored = reranker.rerank(query, scored).await?;
            } else {
                 warn!("Reranking requested but no reranker configured");
            }
        }
        
        // Final sort and limit
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        Ok(SearchResult { results: scored })
    }

    /// Get a memory by ID
    pub async fn get(&self, id: &str) -> Result<Option<MemoryRecord>, MemoryError> {
        let result = self.vector_store.get(id).await?;
        Ok(result.map(|r| r.to_memory_record()))
    }

    /// Get all memories
    pub async fn get_all(&self, options: GetAllOptions) -> Result<Vec<MemoryRecord>, MemoryError> {
        let limit = options.limit.unwrap_or(100);
        let results = self.vector_store.list(None, limit).await?;

        let mut records: Vec<MemoryRecord> =
            results.into_iter().map(|r| r.to_memory_record()).collect();

        // Apply scoping filters
        records.retain(|m| {
            if let Some(ref user_id) = options.user_id {
                if m.user_id.as_ref() != Some(user_id) {
                    return false;
                }
            }
            if let Some(ref agent_id) = options.agent_id {
                if m.agent_id.as_ref() != Some(agent_id) {
                    return false;
                }
            }
            if let Some(ref run_id) = options.run_id {
                if m.run_id.as_ref() != Some(run_id) {
                    return false;
                }
            }
            true
        });

        Ok(records)
    }

    /// Update a memory
    pub async fn update(&self, id: &str, content: &str) -> Result<MemoryRecord, MemoryError> {
        // Get existing record
        let existing = self
            .vector_store
            .get(id)
            .await?
            .ok_or_else(|| MemoryError::NotFound(id.to_string()))?;

        let mut record = existing.to_memory_record();
        let previous_content = record.content.clone();
        record.update_content(content);

        let embedding = self.embedder.embed(content).await?;
        let payload = Payload::from(&record);

        self.vector_store
            .update(id, Some(embedding), payload)
            .await?;

        if let Some(history) = &self.history {
            let _ = history.add_history(
                record.id,
                Some(previous_content),
                record.content.clone(),
                EventType::Update,
                Utc::now(),
                record.user_id.clone(),
                record.agent_id.clone(),
                record.run_id.clone(),
            );
        }

        Ok(record)
    }

    /// Delete a memory
    pub async fn delete(&self, id: &str) -> Result<(), MemoryError> {
        // Get record first for history
        let record = self.get(id).await?;
        
        self.vector_store.delete(id).await?;

        if let Some(record) = record {
            if let Some(history) = &self.history {
                let _ = history.add_history(
                    record.id,
                    Some(record.content),
                    "DELETED".to_string(),
                    EventType::Delete,
                    Utc::now(),
                    record.user_id,
                    record.agent_id,
                    record.run_id,
                );
            }
        }
        
        Ok(())
    }

    /// Get memory history
    pub async fn history(&self, id: &str) -> Result<Vec<HistoryEntry>, MemoryError> {
        if let Some(history) = &self.history {
            let memory_id = Uuid::parse_str(id).map_err(|e| MemoryError::InvalidInput(e.to_string()))?;
            history.get_history(memory_id)
        } else {
            Ok(Vec::new())
        }
    }

    /// Reset all memories
    pub async fn reset(&self, options: ResetOptions) -> Result<(), MemoryError> {
        // Build filters based on options
        let filters = if options.user_id.is_some() || options.agent_id.is_some() {
            // TODO: Build proper filters
            None
        } else {
            None
        };

        self.vector_store.delete_all(filters.as_ref()).await?;
        
        if let Some(history) = &self.history {
            // If global reset, clear history too
            if filters.is_none() {
                history.reset()?;
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_memory_creation() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await;
        assert!(memory.is_ok());
    }

    #[tokio::test]
    async fn test_add_raw() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        let result = memory
            .add(
                "Test memory content",
                AddOptions {
                    user_id: Some("test_user".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().results.len(), 1);
    }

    #[tokio::test]
    async fn test_search() {
        let config = MemoryConfig::default();
        let memory = Memory::new(config).await.unwrap();

        // Add a memory
        memory
            .add(
                "I love programming in Rust",
                AddOptions {
                    user_id: Some("test_user".to_string()),
                    infer: false,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        // Search for it
        let results = memory
            .search(
                "Rust programming",
                SearchOptions {
                    user_id: Some("test_user".to_string()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(!results.results.is_empty());
    }
}
