use async_trait::async_trait;

#[async_trait]
pub trait IntentHandler: Send + Sync {
    async fn handle(&self, query: &str) -> String;
}

pub struct HandlerRegistry {
    handlers: std::collections::HashMap<String, Box<dyn IntentHandler>>,
}

impl HandlerRegistry {
    pub fn new() -> Self {
        Self { handlers: std::collections::HashMap::new() }
    }
    pub fn register(&mut self, intent: &str, handler: Box<dyn IntentHandler>) {
        self.handlers.insert(intent.to_string(), handler);
    }
    pub fn get(&self, intent: &str) -> Option<&Box<dyn IntentHandler>> {
        self.handlers.get(intent)
    }
}
