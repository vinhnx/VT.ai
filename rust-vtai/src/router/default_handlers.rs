use super::handler::{IntentHandler};
use async_trait::async_trait;
use crate::tools::code::run_python_code;

pub struct GeneralConversationHandler;
#[async_trait]
impl IntentHandler for GeneralConversationHandler {
    async fn handle(&self, query: &str) -> String {
        format!("[General] You said: {}", query)
    }
}

pub struct ImageGenerationHandler;
#[async_trait]
impl IntentHandler for ImageGenerationHandler {
    async fn handle(&self, query: &str) -> String {
        format!("[ImageGen] Generating image for: {}", query)
    }
}

pub struct ThinkingModeHandler;
#[async_trait]
impl IntentHandler for ThinkingModeHandler {
    async fn handle(&self, query: &str) -> String {
        // Simulate step-by-step reasoning
        let steps = vec![
            format!("Step 1: Understand the question: {}", query),
            "Step 2: Gather relevant information.".to_string(),
            "Step 3: Analyze and reason through the problem.".to_string(),
            "Step 4: Formulate the answer.".to_string(),
        ];
        steps.join("\n")
    }
}

pub struct CodeAssistanceHandler;
#[async_trait]
impl IntentHandler for CodeAssistanceHandler {
    async fn handle(&self, query: &str) -> String {
        // Example: If the query starts with 'run:', treat the rest as Python code
        if let Some(code) = query.strip_prefix("run:") {
            match run_python_code(code.trim()) {
                Ok(output) => format!("[Code Output]\n{}", output),
                Err(e) => format!("[Code Error]\n{}", e),
            }
        } else {
            format!("[Code] Assisting with code: {}", query)
        }
    }
}
