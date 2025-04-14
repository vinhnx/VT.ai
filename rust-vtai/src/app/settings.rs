use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::utils::models::Settings;
use crate::utils::constants;
use crate::utils::error::Result;
use crate::utils::models_map::MODEL_ALIAS_MAP;

/// Settings update operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsUpdate {
    /// Key being updated
    pub key: String,

    /// New value
    pub value: serde_json::Value,
}

/// Session settings manager
pub struct SettingsManager {
    /// Map of user session IDs to settings
    sessions: Arc<Mutex<HashMap<String, Settings>>>,
}

impl SettingsManager {
    /// Create a new settings manager
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get settings for a session
    pub async fn get_settings(&self, session_id: &str) -> Settings {
        let sessions = self.sessions.lock().await;
        sessions.get(session_id).cloned().unwrap_or_default()
    }

    /// Update settings for a session
    pub async fn update_settings(&self, session_id: &str, update: &SettingsUpdate) -> Result<Settings> {
        let mut sessions = self.sessions.lock().await;

        // Get or create settings for this session
        let settings = sessions.entry(session_id.to_string()).or_insert(Settings::default());

        // Update the appropriate setting
        match update.key.as_str() {
            k if k == constants::SETTINGS_CHAT_MODEL => {
                if let Some(model) = update.value.as_str() {
                    settings.chat_model = model.to_string();
                }
            },
            k if k == constants::SETTINGS_TEMPERATURE => {
                if let Some(temp) = update.value.as_f64() {
                    settings.temperature = temp as f32;
                }
            },
            k if k == constants::SETTINGS_TOP_P => {
                if let Some(top_p) = update.value.as_f64() {
                    settings.top_p = top_p as f32;
                }
            },
            k if k == constants::SETTINGS_IMAGE_GEN => {
                if let Some(enabled) = update.value.as_bool() {
                    settings.image_gen = enabled;
                }
            },
            k if k == constants::SETTINGS_TTS => {
                if let Some(enabled) = update.value.as_bool() {
                    settings.tts = enabled;
                }
            },
            _ => {
                // Ignore unknown settings
            }
        }

        Ok(settings.clone())
    }

    /// Build setting options for UI
    pub fn build_setting_options() -> serde_json::Value {
        let model_options: Vec<_> = MODEL_ALIAS_MAP.iter().map(|(label, value)| {
            serde_json::json!({
                "value": value,
                "label": label,
                // Optionally, you can add icons based on model name or type
            })
        }).collect();

        serde_json::json!({
            "sections": [
                {
                    "name": "Models & Parameters",
                    "items": [
                        {
                            "id": constants::SETTINGS_CHAT_MODEL,
                            "name": "Chat Model",
                            "type": "select",
                            "options": model_options,
                            "initial": constants::DEFAULT_MODEL,
                        },
                        {
                            "id": constants::SETTINGS_TEMPERATURE,
                            "name": "Temperature",
                            "type": "slider",
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.1,
                            "initial": constants::DEFAULT_TEMPERATURE,
                        },
                        {
                            "id": constants::SETTINGS_TOP_P,
                            "name": "Top P",
                            "type": "slider",
                            "min": 0.0,
                            "max": 1.0,
                            "step": 0.1,
                            "initial": constants::DEFAULT_TOP_P,
                        }
                    ]
                },
                {
                    "name": "Features",
                    "items": [
                        {
                            "id": constants::SETTINGS_IMAGE_GEN,
                            "name": "Image Generation",
                            "type": "switch",
                            "initial": false,
                        },
                        {
                            "id": constants::SETTINGS_TTS,
                            "name": "Text-to-Speech",
                            "type": "switch",
                            "initial": false,
                        }
                    ]
                }
            ]
        })
    }
}