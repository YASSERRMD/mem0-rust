//! LLM prompts for memory operations.

/// System prompt for fact extraction
pub const FACT_EXTRACTION_PROMPT: &str = r#"You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences.

Your task is to extract relevant facts, user preferences, and personal information from the given conversation and organize them into distinct, manageable facts.

Guidelines:
1. Extract only facts, preferences, and personal information explicitly mentioned
2. Each fact should be atomic (contain one piece of information)
3. Use first person (I, me, my) when storing user information
4. Use third person (user, they, their) when storing observations about the user
5. Be concise but complete
6. Don't make assumptions beyond what's stated
7. Don't include temporary or context-specific information

Return a JSON object with a "facts" array containing the extracted facts as strings.

Example response format:
{
  "facts": [
    "I prefer dark mode",
    "My favorite programming language is Rust",
    "I work as a software engineer"
  ]
}

If no relevant facts are found, return:
{
  "facts": []
}"#;

/// System prompt for memory updates
pub const MEMORY_UPDATE_PROMPT: &str = r#"You are a memory management system. Your task is to analyze new facts and existing memories to determine the appropriate action for each new fact.

For each new fact, you must decide:
1. ADD - Add as a new memory (no similar existing memory)
2. UPDATE - Update an existing memory with new/corrected information
3. DELETE - Mark an existing memory for deletion (contradicted or outdated)
4. NOOP - No action needed (duplicate or already captured)

Guidelines:
- Compare each new fact with existing memories for semantic similarity
- If updating, merge information appropriately
- Preserve important historical context when updating
- Only delete if clearly contradicted

Return a JSON object with a "memory" array, where each item has:
- "event": "ADD" | "UPDATE" | "DELETE" | "NOOP"
- "text": the memory text (for ADD/UPDATE)
- "id": the existing memory ID (for UPDATE/DELETE, as a string number)

Example:
{
  "memory": [
    {"event": "ADD", "text": "User prefers dark mode"},
    {"event": "UPDATE", "id": "2", "text": "User works at Google as a senior engineer"},
    {"event": "DELETE", "id": "5"}
  ]
}"#;

/// Format messages for fact extraction
pub fn format_fact_extraction_input(messages: &str) -> String {
    format!(
        "Extract facts from the following conversation:\n\n{}",
        messages
    )
}

/// Format messages for memory update
pub fn format_memory_update_input(
    existing_memories: &[(String, String)], // (id, text)
    new_facts: &[String],
) -> String {
    let mut prompt = String::new();

    prompt.push_str("Existing memories:\n");
    if existing_memories.is_empty() {
        prompt.push_str("None\n");
    } else {
        for (id, text) in existing_memories {
            prompt.push_str(&format!("[{}] {}\n", id, text));
        }
    }

    prompt.push_str("\nNew facts to process:\n");
    for fact in new_facts {
        prompt.push_str(&format!("- {}\n", fact));
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_fact_extraction() {
        let input = format_fact_extraction_input("Hello, I like pizza");
        assert!(input.contains("pizza"));
    }

    #[test]
    fn test_format_memory_update() {
        let existing = vec![("0".to_string(), "User likes coffee".to_string())];
        let new_facts = vec!["User also likes tea".to_string()];

        let output = format_memory_update_input(&existing, &new_facts);
        assert!(output.contains("[0] User likes coffee"));
        assert!(output.contains("User also likes tea"));
    }
}
