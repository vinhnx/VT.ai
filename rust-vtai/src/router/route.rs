use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::utils::error::{Result, VTError};
use crate::utils::logger;
use crate::router::constants::{
    INTENT_KEYWORDS, TOPIC_KEYWORDS,
    LAYER_INTENT, LAYER_TOPIC, LAYER_COMPLEXITY, LAYER_SENTIMENT,
    INTENT_CHAT, TOPIC_GENERAL, COMPLEXITY_SIMPLE, SENTIMENT_NEUTRAL,
};

/// Route information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    /// Intent classification
    pub intent: String,

    /// Topic classification
    pub topic: String,

    /// Complexity classification
    pub complexity: String,

    /// Sentiment classification
    pub sentiment: String,
}

impl Default for Route {
    fn default() -> Self {
        Self {
            intent: INTENT_CHAT.to_string(),
            topic: TOPIC_GENERAL.to_string(),
            complexity: COMPLEXITY_SIMPLE.to_string(),
            sentiment: SENTIMENT_NEUTRAL.to_string(),
        }
    }
}

/// Simple keyword-based router
pub struct KeywordRouter;

impl KeywordRouter {
    /// Create a new router
    pub fn new() -> Self {
        KeywordRouter
    }

    /// Route a message to get intent, topic, complexity, and sentiment
    pub fn route(&self, message: &str) -> Result<Route> {
        let message = message.to_lowercase();

        // Determine intent
        let intent = self.classify_intent(&message);

        // Determine topic
        let topic = self.classify_topic(&message);

        // Use simple heuristics for complexity
        let complexity = if message.len() > 500 {
            "complex".to_string()
        } else if message.len() > 100 {
            "moderate".to_string()
        } else {
            "simple".to_string()
        };

        // Default to neutral sentiment for now
        let sentiment = "neutral".to_string();

        logger::info(&format!("Routed message: intent={}, topic={}", intent, topic));

        Ok(Route {
            intent,
            topic,
            complexity,
            sentiment,
        })
    }

    /// Classify intent based on keywords
    fn classify_intent(&self, message: &str) -> String {
        let mut scores: HashMap<&str, usize> = HashMap::new();

        // Score each intent based on keyword matches
        for (intent, keywords) in INTENT_KEYWORDS.iter() {
            let mut score = 0;
            for keyword in keywords {
                if message.contains(keyword) {
                    score += 1;
                }
            }
            scores.insert(intent, score);
        }

        // Return the intent with the highest score, or default
        scores.into_iter()
            .max_by_key(|(_, score)| *score)
            .filter(|(_, score)| *score > 0)
            .map(|(intent, _)| intent.to_string())
            .unwrap_or_else(|| INTENT_CHAT.to_string())
    }

    /// Classify topic based on keywords
    fn classify_topic(&self, message: &str) -> String {
        let mut scores: HashMap<&str, usize> = HashMap::new();

        // Score each topic based on keyword matches
        for (topic, keywords) in TOPIC_KEYWORDS.iter() {
            let mut score = 0;
            for keyword in keywords {
                if message.contains(keyword) {
                    score += 1;
                }
            }
            scores.insert(topic, score);
        }

        // Return the topic with the highest score, or default
        scores.into_iter()
            .max_by_key(|(_, score)| *score)
            .filter(|(_, score)| *score > 0)
            .map(|(topic, _)| topic.to_string())
            .unwrap_or_else(|| TOPIC_GENERAL.to_string())
    }
}