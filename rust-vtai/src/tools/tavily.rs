use reqwest::Client;
use serde_json::Value;
use crate::utils::models::AppConfig;

pub async fn tavily_search(query: &str, config: &AppConfig) -> Result<String, String> {
    let api_key = config.tavily_api_key.as_ref().ok_or("TAVILY_API_KEY not set")?;
    let client = Client::new();
    let url = "https://api.tavily.com/search";
    let payload = serde_json::json!({
        "query": query,
        "api_key": api_key,
    });
    let resp = client.post(url)
        .json(&payload)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(json.to_string())
}
