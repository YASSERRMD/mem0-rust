use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::errors::MemoryError;

use super::models::{GraphEdge, GraphNode};
use super::traits::GraphMemory;

#[derive(Default)]
pub struct InMemoryGraph {
    nodes: RwLock<HashMap<Uuid, GraphNode>>,
    edges: RwLock<HashMap<Uuid, GraphEdge>>,
}

#[async_trait]
impl GraphMemory for InMemoryGraph {
    async fn add_node(&self, node: GraphNode) -> Result<GraphNode, MemoryError> {
        self.nodes.write().await.insert(node.id, node.clone());
        Ok(node)
    }

    async fn add_edge(&self, edge: GraphEdge) -> Result<GraphEdge, MemoryError> {
        let nodes = self.nodes.read().await;
        if !nodes.contains_key(&edge.source) || !nodes.contains_key(&edge.target) {
            return Err(MemoryError::InvalidInput(
                "Edge source/target nodes must exist".to_string(),
            ));
        }
        drop(nodes);

        self.edges.write().await.insert(edge.id, edge.clone());
        Ok(edge)
    }

    async fn get_node(&self, id: &str) -> Result<Option<GraphNode>, MemoryError> {
        let uuid = Uuid::parse_str(id)
            .map_err(|e| MemoryError::InvalidInput(format!("Invalid node id: {e}")))?;
        Ok(self.nodes.read().await.get(&uuid).cloned())
    }

    async fn neighbors(&self, node_id: &str, relation: Option<&str>) -> Result<Vec<GraphNode>, MemoryError> {
        let node_uuid = Uuid::parse_str(node_id)
            .map_err(|e| MemoryError::InvalidInput(format!("Invalid node id: {e}")))?;

        let edges = self.edges.read().await;
        let nodes = self.nodes.read().await;

        let related = edges
            .values()
            .filter(|edge| edge.source == node_uuid)
            .filter(|edge| relation.map_or(true, |r| edge.relation == r))
            .filter_map(|edge| nodes.get(&edge.target).cloned())
            .collect();

        Ok(related)
    }

    async fn list_nodes(&self, limit: usize) -> Result<Vec<GraphNode>, MemoryError> {
        Ok(self.nodes.read().await.values().take(limit).cloned().collect())
    }

    async fn list_edges(&self, limit: usize) -> Result<Vec<GraphEdge>, MemoryError> {
        Ok(self.edges.read().await.values().take(limit).cloned().collect())
    }

    async fn delete_node(&self, id: &str) -> Result<(), MemoryError> {
        let uuid = Uuid::parse_str(id)
            .map_err(|e| MemoryError::InvalidInput(format!("Invalid node id: {e}")))?;

        self.nodes.write().await.remove(&uuid);
        self.edges
            .write()
            .await
            .retain(|_, edge| edge.source != uuid && edge.target != uuid);
        Ok(())
    }

    async fn delete_edge(&self, id: &str) -> Result<(), MemoryError> {
        let uuid = Uuid::parse_str(id)
            .map_err(|e| MemoryError::InvalidInput(format!("Invalid edge id: {e}")))?;
        self.edges.write().await.remove(&uuid);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::{GraphEdge, GraphMemory, GraphNode, GraphNodeKind, InMemoryGraph};

    #[tokio::test]
    async fn creates_and_queries_neighbors() {
        let graph = InMemoryGraph::default();
        let alice = graph
            .add_node(GraphNode::new(GraphNodeKind::Entity, "Alice"))
            .await
            .unwrap();
        let rust = graph
            .add_node(GraphNode::new(GraphNodeKind::Concept, "Rust"))
            .await
            .unwrap();

        graph
            .add_edge(GraphEdge::new(alice.id, rust.id, "likes", 1.0))
            .await
            .unwrap();

        let neighbors = graph.neighbors(&alice.id.to_string(), Some("likes")).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].value, "Rust");
    }
}
