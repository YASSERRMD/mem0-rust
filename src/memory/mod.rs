//! Core memory management module.
//!
//! This module provides:
//! - Memory struct for managing memories
//! - Fact extraction from messages
//! - History tracking

mod manager;
mod prompts;

pub use manager::Memory;
pub use prompts::{FACT_EXTRACTION_PROMPT, MEMORY_UPDATE_PROMPT};
