use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Router layer types for classification
pub const LAYER_INTENT: &str = "intent";
pub const LAYER_TOPIC: &str = "topic";
pub const LAYER_COMPLEXITY: &str = "complexity";
pub const LAYER_SENTIMENT: &str = "sentiment";

/// Intent categories
pub const INTENT_QUERY: &str = "query";
pub const INTENT_COMMAND: &str = "command";
pub const INTENT_CODE: &str = "code";
pub const INTENT_SUMMARIZE: &str = "summarize";
pub const INTENT_GENERATE: &str = "generate";
pub const INTENT_ANALYZE: &str = "analyze";
pub const INTENT_EXTRACT: &str = "extract";
pub const INTENT_SEARCH: &str = "search";
pub const INTENT_EXPLAIN: &str = "explain";
pub const INTENT_CHAT: &str = "chat";

/// Topic categories
pub const TOPIC_GENERAL: &str = "general";
pub const TOPIC_CODING: &str = "coding";
pub const TOPIC_MATH: &str = "math";
pub const TOPIC_SCIENCE: &str = "science";
pub const TOPIC_BUSINESS: &str = "business";
pub const TOPIC_CREATIVE: &str = "creative";
pub const TOPIC_RESEARCH: &str = "research";

/// Complexity levels
pub const COMPLEXITY_SIMPLE: &str = "simple";
pub const COMPLEXITY_MODERATE: &str = "moderate";
pub const COMPLEXITY_COMPLEX: &str = "complex";

/// Sentiment categories
pub const SENTIMENT_NEUTRAL: &str = "neutral";
pub const SENTIMENT_POSITIVE: &str = "positive";
pub const SENTIMENT_NEGATIVE: &str = "negative";
pub const SENTIMENT_URGENT: &str = "urgent";

/// Default embedding model
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-ada-002";

/// Intent keywords mapping for simple classification
pub static INTENT_KEYWORDS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut map = HashMap::new();

    map.insert(INTENT_QUERY, vec!["what", "how", "why", "when", "where", "who", "which", "can you tell me"]);
    map.insert(INTENT_COMMAND, vec!["do", "execute", "run", "perform", "start", "stop", "create"]);
    map.insert(INTENT_CODE, vec!["code", "function", "class", "program", "script", "implement"]);
    map.insert(INTENT_SUMMARIZE, vec!["summarize", "summary", "brief", "overview", "recap", "tldr"]);
    map.insert(INTENT_GENERATE, vec!["generate", "create", "make", "produce", "build", "develop"]);
    map.insert(INTENT_ANALYZE, vec!["analyze", "examine", "evaluate", "assess", "review", "investigate"]);
    map.insert(INTENT_EXTRACT, vec!["extract", "pull", "get", "retrieve", "find"]);
    map.insert(INTENT_SEARCH, vec!["search", "find", "look for", "locate", "seek"]);
    map.insert(INTENT_EXPLAIN, vec!["explain", "clarify", "elaborate", "describe", "define"]);
    map.insert(INTENT_CHAT, vec!["chat", "talk", "converse", "discuss", "tell"]);

    map
});

/// Topic keywords mapping for simple classification
pub static TOPIC_KEYWORDS: Lazy<HashMap<&'static str, Vec<&'static str>>> = Lazy::new(|| {
    let mut map = HashMap::new();

    map.insert(TOPIC_GENERAL, vec!["general", "basic", "simple", "normal"]);
    map.insert(TOPIC_CODING, vec!["code", "programming", "software", "development", "python", "rust", "javascript"]);
    map.insert(TOPIC_MATH, vec!["math", "mathematics", "algebra", "calculus", "geometry", "statistics"]);
    map.insert(TOPIC_SCIENCE, vec!["science", "physics", "chemistry", "biology", "astronomy", "geology"]);
    map.insert(TOPIC_BUSINESS, vec!["business", "finance", "marketing", "management", "economics", "accounting"]);
    map.insert(TOPIC_CREATIVE, vec!["creative", "art", "music", "writing", "design", "poetry"]);
    map.insert(TOPIC_RESEARCH, vec!["research", "academic", "study", "literature", "paper", "journal"]);

    map
});