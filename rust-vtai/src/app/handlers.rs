use std::sync::Arc;
use axum::{
    response::{Html, IntoResponse, Response},
    extract::{Extension, WebSocketUpgrade},
    http::StatusCode,
};
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

use crate::utils::logger;
use crate::app::server::AppState;

/// Handler for the root path
pub async fn index() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <title>VT.ai - Rust Version</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
            }
            .info {
                background-color: #f0f0f0;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>VT.ai - Rust Version</h1>
        <p>Welcome to the Rust implementation of VT.ai!</p>
        <div class="info">
            <p>This is a minimal multimodal AI chat app with dynamic conversation routing.</p>
            <p>Connect using a WebSocket client to interact with the chat API.</p>
        </div>
    </body>
    </html>
    "#)
}

/// Handler for health checks
pub async fn health_check() -> (StatusCode, &'static str) {
    (StatusCode::OK, "OK")
}

/// Message types for WebSocket communication
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum ChatMessage {
    /// Message from user to assistant
    UserMessage(String),

    /// File upload information
    FileUpload { id: String, name: String },

    /// Settings update
    SettingsUpdate(serde_json::Value),

    /// Assistant response
    AssistantResponse(String),

    /// Error message
    Error(String),

    /// Thinking/streaming indicator
    Thinking(bool),

    /// Session initialization
    SessionInit(serde_json::Value),
}

/// Handler for WebSocket chat connections
pub async fn chat_websocket(
    ws: WebSocketUpgrade,
    Extension(state): Extension<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// Handle an individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    logger::info("New WebSocket connection established");

    // Split the socket into sender and receiver
    let (mut sender, mut receiver) = socket.split();

    // Create a channel for sending messages to the WebSocket
    let (tx, mut rx) = mpsc::channel::<Message>(100);

    // Spawn a task to forward messages from the channel to the WebSocket
    let mut send_task = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if sender.send(message).await.is_err() {
                break;
            }
        }
    });

    // Clone tx for use in the receive task
    let tx2 = tx.clone();

    // Spawn a task to process incoming messages from the WebSocket
    let receive_task = tokio::spawn(async move {
        while let Some(Ok(message)) = receiver.next().await {
            match message {
                Message::Text(text) => {
                    // Try to parse the message as a ChatMessage
                    match serde_json::from_str::<ChatMessage>(&text) {
                        Ok(chat_message) => {
                            // Process the message based on its type
                            match chat_message {
                                ChatMessage::UserMessage(content) => {
                                    logger::info(&format!("Received user message: {}", content));

                                    // TODO: Process the user message with LLM and send response
                                    // For now, just echo back
                                    let response = ChatMessage::AssistantResponse(
                                        format!("Echo: {}", content)
                                    );

                                    if let Ok(json) = serde_json::to_string(&response) {
                                        let _ = tx2.send(Message::Text(json)).await;
                                    }
                                },
                                ChatMessage::SettingsUpdate(settings) => {
                                    logger::info(&format!("Received settings update: {:?}", settings));
                                    // TODO: Update user settings
                                },
                                ChatMessage::FileUpload { id, name } => {
                                    logger::info(&format!("Received file upload: {} ({})", name, id));
                                    // TODO: Process file upload
                                },
                                _ => {
                                    logger::warn(&format!("Unhandled message type: {:?}", chat_message));
                                }
                            }
                        },
                        Err(e) => {
                            logger::error(&format!("Failed to parse message: {}", e));
                            let error_message = ChatMessage::Error("Invalid message format".to_string());
                            if let Ok(json) = serde_json::to_string(&error_message) {
                                let _ = tx2.send(Message::Text(json)).await;
                            }
                        }
                    }
                },
                Message::Binary(_) => {
                    logger::info("Received binary message");
                    // TODO: Handle binary messages (e.g., file uploads)
                },
                Message::Close(_) => {
                    logger::info("Client disconnected");
                    break;
                },
                _ => {}
            }
        }
    });

    // Wait for either task to finish
    tokio::select! {
        _ = &mut send_task => receive_task.abort(),
        _ = receive_task => send_task.abort(),
    }

    logger::info("WebSocket connection closed");
}