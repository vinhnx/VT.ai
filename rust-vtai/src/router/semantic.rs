use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, Deserialize)]
pub struct Intent {
    pub name: String,
    pub description: String,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct IntentConfig {
    pub intents: Vec<Intent>,
}

pub struct SemanticRouter {
    pub intents: Vec<Intent>,
}

impl SemanticRouter {
    pub fn from_file(path: &str) -> Self {
        let data = fs::read_to_string(path).expect("Failed to read intents file");
        let config: IntentConfig = serde_json::from_str(&data).expect("Failed to parse intents JSON");
        Self { intents: config.intents }
    }

    // Placeholder: Use simple keyword matching for now
    pub fn classify(&self, query: &str) -> Option<&Intent> {
        for intent in &self.intents {
            for example in &intent.examples {
                if query.to_lowercase().contains(&example.to_lowercase()) {
                    return Some(intent);
                }
            }
        }
        // Fallback: return the first intent
        self.intents.first()
    }
}
