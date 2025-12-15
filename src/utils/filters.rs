//! Metadata filter utilities.

use crate::models::{FilterCondition, FilterLogic, FilterOperator, Filters};

/// Builder for creating filters
pub struct FilterBuilder {
    conditions: Vec<FilterCondition>,
    logic: FilterLogic,
}

impl FilterBuilder {
    /// Create a new filter builder with AND logic
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
            logic: FilterLogic::And,
        }
    }

    /// Create a new filter builder with OR logic
    pub fn new_or() -> Self {
        Self {
            conditions: Vec::new(),
            logic: FilterLogic::Or,
        }
    }

    /// Add an equality condition
    pub fn eq(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Eq,
            value: value.into(),
        });
        self
    }

    /// Add a not-equal condition
    pub fn ne(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Ne,
            value: value.into(),
        });
        self
    }

    /// Add a greater-than condition
    pub fn gt(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Gt,
            value: value.into(),
        });
        self
    }

    /// Add a greater-than-or-equal condition
    pub fn gte(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Gte,
            value: value.into(),
        });
        self
    }

    /// Add a less-than condition
    pub fn lt(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Lt,
            value: value.into(),
        });
        self
    }

    /// Add a less-than-or-equal condition
    pub fn lte(mut self, field: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Lte,
            value: value.into(),
        });
        self
    }

    /// Add an "in list" condition
    pub fn r#in(mut self, field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::In,
            value: serde_json::Value::Array(values),
        });
        self
    }

    /// Add a "not in list" condition
    pub fn nin(mut self, field: impl Into<String>, values: Vec<serde_json::Value>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Nin,
            value: serde_json::Value::Array(values),
        });
        self
    }

    /// Add a contains condition (case-sensitive)
    pub fn contains(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::Contains,
            value: serde_json::Value::String(value.into()),
        });
        self
    }

    /// Add a contains condition (case-insensitive)
    pub fn icontains(mut self, field: impl Into<String>, value: impl Into<String>) -> Self {
        self.conditions.push(FilterCondition {
            field: field.into(),
            operator: FilterOperator::IContains,
            value: serde_json::Value::String(value.into()),
        });
        self
    }

    /// Build the filters
    pub fn build(self) -> Filters {
        Filters {
            conditions: self.conditions,
            logic: self.logic,
        }
    }
}

impl Default for FilterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_builder() {
        let filters = FilterBuilder::new()
            .eq("category", serde_json::json!("work"))
            .gt("priority", serde_json::json!(5))
            .build();

        assert_eq!(filters.conditions.len(), 2);
        assert_eq!(filters.logic, FilterLogic::And);
    }

    #[test]
    fn test_filter_builder_or() {
        let filters = FilterBuilder::new_or()
            .eq("status", serde_json::json!("active"))
            .eq("status", serde_json::json!("pending"))
            .build();

        assert_eq!(filters.logic, FilterLogic::Or);
    }
}
