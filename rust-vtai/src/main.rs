use std::env;
use clap::{Parser, Subcommand};

mod utils;
mod router;
mod tools;
mod assistants;

mod app;

use crate::app::server;
use crate::utils::models::AppConfig;
use crate::utils::logger;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// API keys in format provider=key (e.g., openai=sk-xxx)
    #[arg(long = "api-key")]
    api_keys: Vec<String>,

    /// Model to use for chat (o3-mini, o4, sonnet, haiku, opus, deepseek)
    #[arg(long, default_value = "o3-mini")]
    model: String,

    /// Temperature setting (0.0 to 1.0)
    #[arg(long)]
    temperature: Option<f32>,

    /// Top-p setting (0.0 to 1.0)
    #[arg(long)]
    top_p: Option<f32>,

    /// Port to run the server on
    #[arg(long, default_value = "8000")]
    port: u16,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a one-off query without starting the web server
    Query {
        /// The query text
        #[arg(required = true)]
        text: String,
    },
}

#[tokio::main]
async fn main() {
    // Initialize logger
    logger::init();

    // Load configuration (could be from file or env)
    let config = AppConfig::default();

    // Start the server
    if let Err(e) = server::start_server(config).await {
        eprintln!("Server error: {}", e);
    }
}
