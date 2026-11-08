mod forge;

use std::sync::Mutex;
use serde_json::{json, Value};
use tauri::State;
use vxg_tensor::server::{snapshot_to_json, EmbeddedServer, EmbeddedServerHandle};

use crate::forge::{
    convert_tensor_to_vxt, create_demo_safetensors, default_demo_vxt_path, discover_local_files,
    inspect_tensor_file, ConvertReport, InspectReport,
};

struct AppState {
    _server: Mutex<EmbeddedServer>,
    handle: EmbeddedServerHandle,
}

#[tauri::command]
fn get_studio_state(state: State<'_, AppState>) -> Value {
    let snap = state.handle.snapshot();
    let mut v = snapshot_to_json(&snap);
    if let Some(obj) = v.as_object_mut() {
        obj.insert("discovered_files".into(), json!(discover_local_files()));
    }
    v
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
        Ok(snapshot_to_json(&h.snapshot()))
    })
    .await
    .map_err(|e| e.to_string())?
}

#[tauri::command]
fn unload_model_cmd(state: State<'_, AppState>) -> Result<Value, String> {
    state.handle.unload_model().map_err(|e| e.to_string())?;
    Ok(snapshot_to_json(&state.handle.snapshot()))
}

#[tauri::command]
fn reset_chat_cmd(state: State<'_, AppState>) -> Result<Value, String> {
    state.handle.reset_cache().map_err(|e| e.to_string())?;
    Ok(snapshot_to_json(&state.handle.snapshot()))
}

#[tauri::command]
fn abort_chat_cmd(state: State<'_, AppState>) -> Value {
    state.handle.abort();
    json!({ "ok": true })
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
    input_path: String,
    output_path: String,
    mode: String,
    pack_as_llm: bool,
) -> Result<ConvertReport, String> {
    tauri::async_runtime::spawn_blocking(move || {
        convert_tensor_to_vxt(&input_path, &output_path, &mode, pack_as_llm)
            .map_err(|e| e.to_string())
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
        })
        .invoke_handler(tauri::generate_handler![
            get_studio_state,
            load_model_cmd,
            unload_model_cmd,
            reset_chat_cmd,
            abort_chat_cmd,
            inspect_file_cmd,
            convert_vxt_cmd,
            create_demo_tensor_cmd,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VXT Studio");
}
