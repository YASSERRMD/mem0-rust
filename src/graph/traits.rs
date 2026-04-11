use async_trait::async_trait;

use crate::errors::MemoryError;

use super::models::{GraphEdge, GraphNode};

#[async_trait]
pub trait GraphMemory: Send + Sync {
    async fn add_node(&self, node: GraphNode) -> Result<GraphNode, MemoryError>;
    async fn add_edge(&self, edge: GraphEdge) -> Result<GraphEdge, MemoryError>;

    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, MemoryError>;
    async fn neighbors(&self, node_id: &str, relation: Option<&str>) -> Result<Vec<GraphNode>, MemoryError>;

    async fn list_nodes(&self, limit: usize) -> Result<Vec<GraphNode>, MemoryError>;
    async fn list_edges(&self, limit: usize) -> Result<Vec<GraphEdge>, MemoryError>;

    async fn delete_node(&self, id: &str) -> Result<(), MemoryError>;
    async fn delete_edge(&self, id: &str) -> Result<(), MemoryError>;
}
