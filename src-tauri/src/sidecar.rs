//! Manages the Python sidecar subprocess.
//!
//! The sidecar is a long-running `python -m beet.sidecar` (or the PyInstaller
//! binary in bundled builds) that speaks line-delimited JSON-RPC over
//! stdin/stdout. This module spawns it, routes requests, and matches responses
//! to callers via request IDs.
//!
//! Dev override: set `BEET_SIDECAR_CMD` (e.g. `python -m beet.sidecar
//! --profile default`) to run the Python source directly instead of the
//! bundled binary. Whitespace-split; do not use paths with spaces.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::oneshot;
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[error("{code}: {message}")]
pub struct SidecarError {
    pub code: String,
    pub message: String,
}

impl SidecarError {
    fn io(msg: impl Into<String>) -> Self {
        Self {
            code: "ERR_IO".into(),
            message: msg.into(),
        }
    }
    fn gone() -> Self {
        Self {
            code: "ERR_SIDECAR_GONE".into(),
            message: "sidecar response channel closed".into(),
        }
    }
}

type PendingMap = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, SidecarError>>>>>;

pub struct SidecarManager {
    child: Mutex<Child>,
    stdin: Mutex<ChildStdin>,
    next_id: AtomicU64,
    pending: PendingMap,
}

impl SidecarManager {
    pub fn spawn(resource_dir: &Path) -> Result<Self> {
        let (program, args) = resolve_sidecar_command(resource_dir)?;
        info!(program = %program, args = ?args, "spawning sidecar");

        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("failed to spawn sidecar: {program}"))?;

        let stdin = child.stdin.take().context("sidecar missing stdin")?;
        let stdout = child.stdout.take().context("sidecar missing stdout")?;

        let pending: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let reader_pending = Arc::clone(&pending);

        thread::Builder::new()
            .name("beet-sidecar-reader".into())
            .spawn(move || reader_loop(stdout, reader_pending))
            .context("failed to spawn sidecar reader thread")?;

        Ok(Self {
            child: Mutex::new(child),
            stdin: Mutex::new(stdin),
            next_id: AtomicU64::new(1),
            pending,
        })
    }

    pub async fn send_request(&self, method: &str, params: Value) -> Result<Value, SidecarError> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().expect("pending map poisoned").insert(id, tx);

        let req = json!({
            "id": id,
            "method": method,
            "params": params,
        });
        let line = serde_json::to_string(&req).expect("serialize request");

        {
            let mut stdin = self.stdin.lock().expect("stdin poisoned");
            writeln!(stdin, "{line}").map_err(|e| SidecarError::io(format!("write failed: {e}")))?;
            stdin.flush().ok();
        }

        rx.await.map_err(|_| SidecarError::gone())?
    }
}

impl Drop for SidecarManager {
    fn drop(&mut self) {
        // Best-effort graceful shutdown: send the shutdown RPC, then kill if
        // the child lingers. We cannot await here, so we write directly and
        // give the child a moment before killing.
        if let Ok(mut stdin) = self.stdin.lock() {
            let _ = writeln!(stdin, "{}", json!({"id": 0, "method": "shutdown"}));
            let _ = stdin.flush();
        }
        if let Ok(mut child) = self.child.lock() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn reader_loop(
    stdout: std::process::ChildStdout,
    pending: PendingMap,
) {
    let reader = BufReader::new(stdout);
    for line in reader.lines() {
        let Ok(line) = line else {
            warn!("sidecar stdout closed");
            break;
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Ok(msg) = serde_json::from_str::<Value>(trimmed) else {
            warn!(line = %trimmed, "sidecar emitted invalid JSON");
            continue;
        };
        // Server-initiated events (e.g. the startup "ready" line) have no
        // request id — drop them here; wiring them to the UI is future work.
        if msg.get("event").is_some() {
            info!(event = %msg, "sidecar event");
            continue;
        }
        let id = msg.get("id").and_then(|v| v.as_u64()).unwrap_or(0);
        let mut map = pending.lock().expect("pending map poisoned");
        let Some(tx) = map.remove(&id) else {
            warn!(id, "sidecar response has no pending waiter");
            continue;
        };
        if let Some(err) = msg.get("error") {
            let parsed: SidecarError = serde_json::from_value(err.clone()).unwrap_or_else(|_| {
                SidecarError {
                    code: "ERR_UNKNOWN".into(),
                    message: err.to_string(),
                }
            });
            let _ = tx.send(Err(parsed));
        } else if let Some(result) = msg.get("result") {
            let _ = tx.send(Ok(result.clone()));
        } else {
            let _ = tx.send(Err(SidecarError {
                code: "ERR_BAD_RESPONSE".into(),
                message: format!("response missing result/error: {msg}"),
            }));
        }
    }

    // Reader exiting means the sidecar died. Fail all in-flight requests so
    // callers don't hang forever. The app remains alive; the user can retry
    // (or relaunch — v0 does not auto-restart).
    error!("sidecar reader exiting; failing pending requests");
    let mut map = pending.lock().expect("pending map poisoned");
    for (_id, tx) in map.drain() {
        let _ = tx.send(Err(SidecarError::gone()));
    }
}

fn resolve_sidecar_command(resource_dir: &Path) -> Result<(String, Vec<String>)> {
    if let Ok(cmd) = std::env::var("BEET_SIDECAR_CMD") {
        let parts: Vec<String> = cmd.split_whitespace().map(String::from).collect();
        if parts.is_empty() {
            anyhow::bail!("BEET_SIDECAR_CMD is empty");
        }
        let program = parts[0].clone();
        let args = parts[1..].to_vec();
        return Ok((program, args));
    }

    let path = sidecar_binary_path(resource_dir);
    if !path.exists() {
        anyhow::bail!(
            "sidecar binary not found at {} and BEET_SIDECAR_CMD not set. \
             For dev, export BEET_SIDECAR_CMD=\"python -m beet.sidecar\". \
             For bundled builds, ensure PyInstaller output is at src-tauri/binaries/.",
            path.display()
        );
    }
    Ok((path.to_string_lossy().into_owned(), Vec::new()))
}

fn sidecar_binary_path(resource_dir: &Path) -> PathBuf {
    let name = if cfg!(windows) { "beet-sidecar.exe" } else { "beet-sidecar" };
    resource_dir.join("binaries").join(name)
}
