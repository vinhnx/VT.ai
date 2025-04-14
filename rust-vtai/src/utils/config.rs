use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use serde_json;

use crate::utils::error::{Result, VTError};
use crate::utils::models::AppConfig;
use crate::utils::logger;

/// Path to the config directory within the user's home directory
const CONFIG_DIR: &str = ".vtai";

/// Config file name
const CONFIG_FILE: &str = "config.json";

/// Find the config directory, creating it if needed
fn get_config_dir() -> Result<PathBuf> {
    let home_dir = dirs::home_dir()
        .ok_or_else(|| VTError::ConfigError("Could not find home directory".to_string()))?;

    let config_dir = home_dir.join(CONFIG_DIR);

    if !config_dir.exists() {
        fs::create_dir_all(&config_dir)
            .map_err(|e| VTError::ConfigError(format!("Failed to create config directory: {}", e)))?;

        logger::info(&format!("Created config directory: {}", config_dir.display()));
    }

    Ok(config_dir)
}

/// Get the path to the config file
fn get_config_file() -> Result<PathBuf> {
    let config_dir = get_config_dir()?;
    Ok(config_dir.join(CONFIG_FILE))
}

/// Load the application configuration
pub fn load_config() -> Result<AppConfig> {
    let config_file = get_config_file()?;

    if !config_file.exists() {
        logger::info("Config file not found, using default configuration");
        return Ok(AppConfig::default());
    }

    let config_data = fs::read_to_string(&config_file)
        .map_err(|e| VTError::ConfigError(format!("Failed to read config file: {}", e)))?;

    let config: AppConfig = serde_json::from_str(&config_data)
        .map_err(|e| VTError::ConfigError(format!("Failed to parse config file: {}", e)))?;

    logger::info(&format!("Loaded configuration from {}", config_file.display()));

    Ok(config)
}

/// Save the application configuration
pub fn save_config(config: &AppConfig) -> Result<()> {
    let config_file = get_config_file()?;

    let config_data = serde_json::to_string_pretty(config)
        .map_err(|e| VTError::ConfigError(format!("Failed to serialize configuration: {}", e)))?;

    fs::write(&config_file, config_data)
        .map_err(|e| VTError::ConfigError(format!("Failed to write config file: {}", e)))?;

    logger::info(&format!("Saved configuration to {}", config_file.display()));

    Ok(())
}

/// Initialize the application with configuration
pub async fn initialize_app() -> Result<AppConfig> {
    // Load the base configuration
    let mut config = load_config()?;

    // Check environment variables for API keys
    if let Ok(openai_key) = env::var("OPENAI_API_KEY") {
        config.openai_api_key = Some(openai_key);
    }

    if let Ok(anthropic_key) = env::var("ANTHROPIC_API_KEY") {
        config.anthropic_api_key = Some(anthropic_key);
    }

    // Look for OpenAI Assistant ID
    if let Ok(assistant_id) = env::var("OPENAI_ASSISTANT_ID") {
        config.assistant_id = Some(assistant_id);
    }

    // Print info about the loaded configuration
    logger::info(&format!("Initialized app with model: {}", config.default_model));
    logger::info(&format!("OpenAI API key configured: {}", config.openai_api_key.is_some()));
    logger::info(&format!("Anthropic API key configured: {}", config.anthropic_api_key.is_some()));

    Ok(config)
}

/// Create a default configuration file if none exists
pub fn ensure_default_config() -> Result<()> {
    let config_file = get_config_file()?;

    if !config_file.exists() {
        let default_config = AppConfig::default();
        save_config(&default_config)?;
        logger::info("Created default configuration file");
    }

    Ok(())
}