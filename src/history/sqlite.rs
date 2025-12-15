use crate::models::{EventType, HistoryEntry};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use crate::errors::MemoryError;

pub struct HistoryManager {
    conn: Arc<Mutex<Connection>>,
}

impl HistoryManager {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, MemoryError> {
        // Ensure directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| MemoryError::History(e.to_string()))?;
        }

        let conn = Connection::open(path).map_err(|e| MemoryError::History(e.to_string()))?;
        
        // Create table
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                previous_content TEXT,
                new_content TEXT NOT NULL,
                event TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                agent_id TEXT,
                run_id TEXT
            )",
            [],
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_history(
        &self,
        memory_id: Uuid,
        previous_content: Option<String>,
        new_content: String,
        event: EventType,
        timestamp: DateTime<Utc>,
        user_id: Option<String>,
        agent_id: Option<String>,
        run_id: Option<String>,
    ) -> Result<(), MemoryError> {
        let conn = self.conn.lock().unwrap();
        let id = Uuid::new_v4().to_string();
        
        // Serialize event enum
        let event_str = serde_json::to_string(&event)
            .map_err(|e| MemoryError::History(format!("Failed to serialize event: {}", e)))?;
        let event_str = event_str.trim_matches('"');

        conn.execute(
            "INSERT INTO history (id, memory_id, previous_content, new_content, event, timestamp, user_id, agent_id, run_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                id,
                memory_id.to_string(),
                previous_content,
                new_content,
                event_str,
                timestamp.to_rfc3339(),
                user_id,
                agent_id,
                run_id,
            ],
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        Ok(())
    }

    pub fn get_history(&self, memory_id: Uuid) -> Result<Vec<HistoryEntry>, MemoryError> {
        let conn = self.conn.lock().unwrap();
        
        let mut stmt = conn.prepare(
            "SELECT id, memory_id, previous_content, new_content, event, timestamp 
             FROM history WHERE memory_id = ?1 ORDER BY timestamp DESC"
        ).map_err(|e| MemoryError::History(e.to_string()))?;

        let rows = stmt.query_map(params![memory_id.to_string()], |row| {
            let event_str: String = row.get(4)?;
            let event = match event_str.as_str() {
                "ADD" => EventType::Add,
                "UPDATE" => EventType::Update,
                "DELETE" => EventType::Delete,
                _ => EventType::Noop,
            };

            let timestamp_str: String = row.get(5)?;
            let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or(Utc::now());

            Ok(HistoryEntry {
                id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                memory_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                previous_content: row.get(2)?,
                new_content: row.get(3)?,
                event,
                timestamp,
            })
        }).map_err(|e| MemoryError::History(e.to_string()))?;

        let mut history = Vec::new();
        for row in rows {
            history.push(row.map_err(|e| MemoryError::History(e.to_string()))?);
        }
        
        Ok(history)
    }
    
    pub fn reset(&self) -> Result<(), MemoryError> {
         let conn = self.conn.lock().unwrap();
         conn.execute("DELETE FROM history", []).map_err(|e| MemoryError::History(e.to_string()))?;
         Ok(())
    }
}
