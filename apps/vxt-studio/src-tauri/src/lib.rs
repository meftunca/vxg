mod forge;
mod hf_hub;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, State};
use vxg_tensor::server::{snapshot_to_json, EmbeddedServer, EmbeddedServerHandle};

use crate::forge::{
    convert_tensor_to_vxt_with_progress, create_demo_safetensors, default_demo_vxt_path,
    discover_local_files, inspect_tensor_file, ConvertReport, InspectReport,
};
use crate::hf_hub::{
    default_hf_vxt_output_path, delete_vxt_model_file, download_hf_repo, search_hf_models,
    ForgeProgressPayload, HfSearchModel,
};

struct AppState {
    _server: Mutex<EmbeddedServer>,
    handle: EmbeddedServerHandle,
    forge_cancel: Arc<AtomicBool>,
}

fn studio_snapshot_with_files(handle: &EmbeddedServerHandle) -> Value {
    let snap = handle.snapshot();
    let mut v = snapshot_to_json(&snap);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("discovered_files".into(), json!(discover_local_files()));
    }
    v
}

#[tauri::command]
async fn get_studio_state(state: State<'_, AppState>) -> Result<Value, String> {
    let h = state.handle.clone();
    tauri::async_runtime::spawn_blocking(move || studio_snapshot_with_files(&h))
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn load_model_cmd(
    state: State<'_, AppState>,
    path: String,
    use_gpu: Option<bool>,
) -> Result<Value, String> {
    let h = state.handle.clone();
    let gpu = use_gpu.unwrap_or(true);
    tauri::async_runtime::spawn_blocking(move || {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        h.load_model(path, gpu, threads).map_err(|e| e.to_string())?;
        Ok(studio_snapshot_with_files(&h))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn set_draft_model_cmd(
    state: State<'_, AppState>,
    path: Option<String>,
    use_gpu: Option<bool>,
) -> Result<Value, String> {
    let h = state.handle.clone();
    let gpu = use_gpu.unwrap_or(true);
    tauri::async_runtime::spawn_blocking(move || {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        h.set_draft_model(path, gpu, threads)
            .map_err(|e| e.to_string())?;
        Ok(studio_snapshot_with_files(&h))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn unload_model_cmd(state: State<'_, AppState>) -> Result<Value, String> {
    let h = state.handle.clone();
    tauri::async_runtime::spawn_blocking(move || {
        h.unload_model().map_err(|e| e.to_string())?;
        Ok(studio_snapshot_with_files(&h))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn reset_chat_cmd(state: State<'_, AppState>) -> Result<Value, String> {
    let h = state.handle.clone();
    tauri::async_runtime::spawn_blocking(move || {
        h.reset_cache().map_err(|e| e.to_string())?;
        Ok(studio_snapshot_with_files(&h))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn abort_chat_cmd(state: State<'_, AppState>) -> Value {
    state.handle.abort();
    json!({ "ok": true })
}

#[tauri::command]
fn cancel_forge_job_cmd(state: State<'_, AppState>) -> Value {
    state.forge_cancel.store(true, Ordering::SeqCst);
    json!({ "ok": true })
}

#[tauri::command]
async fn delete_model_cmd(state: State<'_, AppState>, path: String) -> Result<Value, String> {
    let h = state.handle.clone();
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(loaded) = h.snapshot().model {
            if loaded.model_path == path {
                let _ = h.unload_model();
            }
        }
        delete_vxt_model_file(&path).map_err(|e| e.to_string())?;
        Ok(studio_snapshot_with_files(&h))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn inspect_file_cmd(path: String) -> Result<InspectReport, String> {
    tauri::async_runtime::spawn_blocking(move || {
        inspect_tensor_file(&path).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn convert_vxt_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    input_path: String,
    output_path: String,
    mode: String,
    pack_as_llm: bool,
) -> Result<ConvertReport, String> {
    state.forge_cancel.store(false, Ordering::SeqCst);
    let cancel = state.forge_cancel.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let emit_pack = |done: usize, total: usize, name: &str| {
            let pct = if total > 0 {
                (done as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            let _ = app.emit(
                "vxt-progress",
                ForgeProgressPayload {
                    phase: "pack".into(),
                    current: done as u64,
                    total: total as u64,
                    percent: pct.clamp(0.0, 100.0),
                    detail: format!("Sıkıştırılıyor ({done}/{total}): {name}"),
                },
            );
        };
        convert_tensor_to_vxt_with_progress(
            &input_path,
            &output_path,
            &mode,
            pack_as_llm,
            &cancel,
            &emit_pack,
        )
        .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn download_hf_and_pack_cmd(
    app: AppHandle,
    state: State<'_, AppState>,
    repo_id: String,
    mode: String,
    hf_token: Option<String>,
    output_path: Option<String>,
) -> Result<ConvertReport, String> {
    state.forge_cancel.store(false, Ordering::SeqCst);
    let cancel = state.forge_cancel.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let out_path = match output_path.filter(|s| !s.trim().is_empty()) {
            Some(p) => p,
            None => default_hf_vxt_output_path(&repo_id, &mode)
                .map_err(|e| e.to_string())?
                .to_string_lossy()
                .into_owned(),
        };
        let emit_dl = |p: ForgeProgressPayload| {
            let _ = app.emit("vxt-progress", p);
        };
        let downloaded = download_hf_repo(&repo_id, hf_token.as_deref(), &cancel, &emit_dl)
            .map_err(|e| e.to_string())?;
        let in_str = downloaded.to_string_lossy().into_owned();
        let emit_pack = |done: usize, total: usize, name: &str| {
            let pct = if total > 0 {
                (done as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            let _ = app.emit(
                "vxt-progress",
                ForgeProgressPayload {
                    phase: "pack".into(),
                    current: done as u64,
                    total: total as u64,
                    percent: pct.clamp(0.0, 100.0),
                    detail: format!(".vxt paketleniyor ({done}/{total}): {name}"),
                },
            );
        };
        convert_tensor_to_vxt_with_progress(&in_str, &out_path, &mode, true, &cancel, &emit_pack)
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn search_hf_models_cmd(
    query: String,
    hf_token: Option<String>,
) -> Result<Vec<HfSearchModel>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        search_hf_models(&query, hf_token.as_deref()).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
async fn create_demo_tensor_cmd() -> Result<InspectReport, String> {
    tauri::async_runtime::spawn_blocking(|| {
        create_demo_safetensors().map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server = EmbeddedServer::start("127.0.0.1", 11435)
        .or_else(|_| EmbeddedServer::start("127.0.0.1", 0))
        .expect("failed to bind local VXT OpenAI server");
    let handle = server.handle();

    if let Some(default_vxt) = default_demo_vxt_path() {
        let bg = handle.clone();
        std::thread::spawn(move || {
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
            let _ = bg.load_model(default_vxt, true, threads);
        });
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            _server: Mutex::new(server),
            handle,
            forge_cancel: Arc::new(AtomicBool::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            get_studio_state,
            load_model_cmd,
            set_draft_model_cmd,
            unload_model_cmd,
            reset_chat_cmd,
            abort_chat_cmd,
            cancel_forge_job_cmd,
            delete_model_cmd,
            inspect_file_cmd,
            convert_vxt_cmd,
            download_hf_and_pack_cmd,
            search_hf_models_cmd,
            create_demo_tensor_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VXT Studio");
}
