#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod sidecar;

use std::sync::Arc;

use tauri::Manager;

use crate::sidecar::SidecarManager;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    tauri::Builder::default()
        .setup(|app| {
            let resource_dir = app
                .path()
                .resource_dir()
                .expect("failed to resolve resource_dir");
            let manager = SidecarManager::spawn(&resource_dir)
                .expect("failed to spawn sidecar");
            app.manage(Arc::new(manager));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::analyze,
            commands::analyze_batch,
            commands::feedback,
            commands::health,
            commands::history_list,
            commands::history_get,
            commands::history_delete,
            commands::history_export,
            commands::list_profiles,
            commands::get_config,
            commands::switch_profile,
            commands::monitoring_summary,
            commands::monitoring_timeline,
            commands::monitoring_detectors,
            commands::run_eval,
            commands::monitoring_drift,
            commands::monitoring_set_baseline,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
