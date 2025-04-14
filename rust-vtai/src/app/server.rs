use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::{get, get_service},
    Json, Router,
};
use tower_http::{
    cors::CorsLayer,
    services::ServeDir,
};
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;
use futures::{StreamExt, SinkExt};

use crate::utils::error::{Result, VTError};
use crate::utils::models::{AppConfig, Settings};
use crate::utils::logger;
use crate::assistants::openai::chat_with_openai_with_config;
use crate::assistants::anthropic::chat_with_anthropic;
use crate::assistants::deepseek::chat_with_deepseek;
use crate::assistants::gemini::chat_with_gemini;
use crate::assistants::mistral::chat_with_mistral;
use crate::assistants::cohere::chat_with_cohere;
use crate::tools::file::process_file;
use crate::tools::code::run_python_code;
use crate::utils::models_map::{resolve_model_alias, is_reasoning_model};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WebSocketMessage {
    /// Message type (UserMessage, AssistantResponse, Error, Thinking, SettingsUpdate)
    r#type: String,

    /// Message data
    data: serde_json::Value,
}

/// Chat session state
struct ChatSession {
    /// User settings
    settings: Settings,
}

impl Default for ChatSession {
    fn default() -> Self {
        Self {
            settings: Settings::default(),
        }
    }
}

/// Application state
pub struct AppState {
    /// Application configuration
    config: AppConfig,

    /// Active chat sessions
    sessions: Mutex<Vec<Arc<Mutex<ChatSession>>>>,

    /// Enable or disable dynamic routing
    pub dynamic_routing: bool,
}

/// Start the web server
pub async fn start_server(config: AppConfig) -> Result<()> {
    // Create shared application state
    let state = Arc::new(AppState {
        config,
        sessions: Mutex::new(Vec::new()),
        dynamic_routing: true, // Set to false to disable
    });

    // Get the port number
    let port = state.config.port;

    // Set up static file serving
    let static_path = std::env::current_dir()?.join("static");

    // Create the router
    let app = Router::new()
        // API routes
        .route("/api/health", get(health_handler))
        .route("/api/chat", get(websocket_handler))

        // Static file serving for the web UI
        .nest_service("/", get_service(ServeDir::new(static_path)))

        // Add CORS layer
        .layer(CorsLayer::permissive())

        // Add application state
        .with_state(state);

    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    logger::info(&format!("Starting web server on http://localhost:{}", port));

    // Use tokio::net::TcpListener for axum::serve
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service())
        .await
        .map_err(|e| VTError::Internal(format!("Server error: {}", e)))?;

    Ok(())
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// WebSocket connection handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Create a new chat session
    let session = Arc::new(Mutex::new(ChatSession::default()));

    // Store the session
    state.sessions.lock().await.push(session.clone());

    // Upgrade the connection to a WebSocket
    ws.on_upgrade(move |socket| handle_socket(socket, state, session))
}

/// Handle WebSocket connection
async fn handle_socket(
    socket: WebSocket,
    state: Arc<AppState>,
    session: Arc<Mutex<ChatSession>>,
) {
    logger::info("New WebSocket connection established");

    // Split the socket into sender and receiver
    let (mut sender, mut receiver) = socket.split();

    // Welcome message
    let welcome_msg = WebSocketMessage {
        r#type: "AssistantResponse".to_string(),
        data: json!("Welcome to VT.ai (Rust Edition)! How can I help you today?"),
    };

    // Send welcome message
    if let Err(e) = sender.send(axum::extract::ws::Message::Text(
        serde_json::to_string(&welcome_msg).unwrap()
    )).await {
        logger::error(&format!("Error sending welcome message: {}", e));
        return;
    }

    // Process incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            axum::extract::ws::Message::Text(text) => {
                // Parse the incoming message
                match serde_json::from_str::<WebSocketMessage>(&text) {
                    Ok(ws_msg) => {
                        match ws_msg.r#type.as_str() {
                            "UserMessage" => {
                                // Process user message
                                if let Some(mut message_content) = ws_msg.data.as_str().map(String::from) {
                                    logger::info(&format!("Received user message: {}", message_content));

                                    // Determine which model/provider to use and check settings
                                    let session_guard = session.lock().await;
                                    let model_alias = session_guard.settings.chat_model.clone();
                                    let use_thinking = session_guard.settings.use_thinking_model;
                                    drop(session_guard);

                                    // Resolve alias to actual model ID
                                    let model_id = resolve_model_alias(&model_alias).unwrap_or(&model_alias);

                                    logger::info(&format!("Using model: {} (resolved from alias: {})", model_id, model_alias));

                                    // Add <think> tag if needed
                                    if use_thinking && is_reasoning_model(model_id) && !message_content.to_lowercase().contains("<think>") {
                                        logger::info("Adding <think> tag for reasoning model");
                                        message_content = format!("<think>{}</think>", message_content);
                                    }

                                    let response_text = if model_id.starts_with("gpt-") || model_id.starts_with("o1") || model_id.starts_with("o3-") || model_id.starts_with("openai/") {
                                        chat_with_openai_with_config(&message_content, &state.config).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("claude-") || model_id.starts_with("anthropic/") {
                                        chat_with_anthropic(&message_content).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("deepseek/") {
                                        chat_with_deepseek(&message_content).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("gemini/") {
                                        chat_with_gemini(&message_content).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("openrouter/") {
                                        chat_with_openai_with_config(&message_content, &state.config).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("ollama/") {
                                        chat_with_openai_with_config(&message_content, &state.config).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("mistral/") {
                                        chat_with_mistral(&message_content).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("groq/") {
                                        chat_with_openai_with_config(&message_content, &state.config).await.unwrap_or_else(|e| e.to_string())
                                    } else if model_id.starts_with("cohere/") || model_id.starts_with("command") {
                                        chat_with_cohere(&message_content).await.unwrap_or_else(|e| e.to_string())
                                    } else if message_content.starts_with("/file ") {
                                        let path = message_content.trim_start_matches("/file ");
                                        process_file(path).unwrap_or_else(|e| e)
                                    } else if message_content.starts_with("/code ") {
                                        let code = message_content.trim_start_matches("/code ");
                                        run_python_code(code).unwrap_or_else(|e| e)
                                    } else {
                                        // Fallback or handle other providers/commands
                                        format!("Unknown model provider or command for model: {}", model_id)
                                    };

                                    let response = WebSocketMessage {
                                        r#type: "AssistantResponse".to_string(),
                                        data: json!(response_text),
                                    };

                                    if let Err(e) = sender.send(axum::extract::ws::Message::Text(
                                        serde_json::to_string(&response).unwrap()
                                    )).await {
                                        logger::error(&format!("Error sending response: {}", e));
                                        break;
                                    }
                                }
                            },
                            "SettingsUpdate" => {
                                // Update session settings
                                if let Ok(new_settings) = serde_json::from_value::<Settings>(ws_msg.data) {
                                    let mut session = session.lock().await;
                                    // Update all fields from the incoming settings
                                    session.settings = new_settings.clone();

                                    logger::info(&format!(
                                        "Updated settings: model={}, vision={}, img_model={}, tts_model={}, dynamic_routing={}, thinking_model={}",
                                        new_settings.chat_model,
                                        new_settings.vision_model,
                                        new_settings.image_gen_llm_model,
                                        new_settings.tts_model,
                                        new_settings.use_dynamic_conversation_routing,
                                        new_settings.use_thinking_model
                                    ));

                                    // Confirm settings update
                                    let response = WebSocketMessage {
                                        r#type: "SettingsUpdated".to_string(),
                                        data: json!(new_settings),
                                    };

                                    if let Err(e) = sender.send(axum::extract::ws::Message::Text(
                                        serde_json::to_string(&response).unwrap()
                                    )).await {
                                        logger::error(&format!("Error confirming settings update: {}", e));
                                        break;
                                    }
                                }
                            },
                            _ => {
                                logger::warn(&format!("Unknown message type: {}", ws_msg.r#type));
                            }
                        }
                    },
                    Err(e) => {
                        logger::error(&format!("Error parsing message: {}", e));
                    }
                }
            },
            axum::extract::ws::Message::Close(_) => {
                logger::info("WebSocket connection closed");
                break;
            },
            _ => {}
        }
    }

    logger::info("WebSocket connection ended");
}