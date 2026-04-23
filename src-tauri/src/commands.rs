//! Tauri `#[command]` handlers. Each is a thin bridge to the sidecar.
//!
//! Frontend calls `invoke('<cmd_name>', { ... })`; we forward as JSON-RPC.
//! Errors bubble up as `SidecarError` (serialized to JSON for the frontend).

use std::sync::Arc;

use serde_json::{json, Value};
use tauri::State;

use crate::sidecar::{SidecarError, SidecarManager};

#[tauri::command]
pub async fn analyze(
    text: String,
    submission_id: Option<String>,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager
        .send_request(
            "analyze",
            json!({ "text": text, "submission_id": submission_id }),
        )
        .await
}

#[tauri::command]
pub async fn analyze_batch(
    items: Value,
    batch_id: Option<String>,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager
        .send_request(
            "analyze_batch",
            json!({ "items": items, "batch_id": batch_id }),
        )
        .await
}

#[tauri::command]
pub async fn feedback(
    text: String,
    confirmed_label: i32,
    submission_id: Option<String>,
    reviewer_notes: Option<String>,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    let params = json!({
        "text": text,
        "confirmed_label": confirmed_label,
        "submission_id": submission_id,
        "reviewer_notes": reviewer_notes,
    });
    manager.send_request("feedback", params).await
}

#[tauri::command]
pub async fn health(
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("health", json!({})).await
}

#[tauri::command]
pub async fn history_list(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("history_list", params).await
}

#[tauri::command]
pub async fn history_get(
    submission_id: String,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager
        .send_request("history_get", json!({ "submission_id": submission_id }))
        .await
}

#[tauri::command]
pub async fn history_delete(
    submission_id: String,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager
        .send_request("history_delete", json!({ "submission_id": submission_id }))
        .await
}

#[tauri::command]
pub async fn history_export(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("history_export", params).await
}

#[tauri::command]
pub async fn list_profiles(
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("list_profiles", json!({})).await
}

#[tauri::command]
pub async fn get_config(
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("get_config", json!({})).await
}

#[tauri::command]
pub async fn switch_profile(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("switch_profile", params).await
}

#[tauri::command]
pub async fn monitoring_summary(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("monitoring_summary", params).await
}

#[tauri::command]
pub async fn monitoring_timeline(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("monitoring_timeline", params).await
}

#[tauri::command]
pub async fn monitoring_detectors(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("monitoring_detectors", params).await
}

#[tauri::command]
pub async fn run_eval(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("run_eval", params).await
}

#[tauri::command]
pub async fn monitoring_drift(
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("monitoring_drift", json!({})).await
}

#[tauri::command]
pub async fn monitoring_set_baseline(
    params: Value,
    manager: State<'_, Arc<SidecarManager>>,
) -> Result<Value, SidecarError> {
    manager.send_request("monitoring_set_baseline", params).await
}
