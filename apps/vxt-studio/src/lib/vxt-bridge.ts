import { invoke } from "@tauri-apps/api/core";

export interface LoadedModelInfo {
  path: string;
  name: string;
  arch: string;
  n_layer: number;
  n_embd: number;
  n_vocab: number;
  compressed_bytes: number;
  compressed_mib: number;
  backend: string;
  cb4_fast: boolean;
  turbo_fast: boolean;
  vision?: boolean;
  load_ms: number;
}

export interface TurnMetricsInfo {
  prefill_ms: number;
  decode_ms: number;
  prefill_tps: number;
  decode_tps: number;
  stream_gbs: number;
  prompt_tokens: number;
  cached_tokens: number;
  new_prefill_tokens: number;
  generated_tokens: number;
  finish_reason: string;
  decode_path: string;
}

export interface DiscoveredFile {
  path: string;
  name: string;
  kind: string;
  size_mib: number;
  runnable_hint: boolean;
}

export interface StudioState {
  ok: boolean;
  port: number;
  base_url: string;
  busy: boolean;
  cached_tokens: number;
  model: LoadedModelInfo | null;
  last_metrics: TurnMetricsInfo | null;
  last_error: string | null;
  discovered_files?: DiscoveredFile[];
}

export interface TensorRowInfo {
  index: number;
  name: string;
  shape: string;
  codec_or_dtype: string;
  bpw: number;
  size_kib: number;
}

export interface InspectReport {
  path: string;
  file_name: string;
  format: string;
  size_bytes: number;
  size_mib: number;
  tensor_count: number;
  metadata_count: number;
  runnable_llm: boolean;
  arch_hint: string | null;
  tensors: TensorRowInfo[];
}

export interface ConvertReport {
  input_path: string;
  output_path: string;
  mode: string;
  runnable_llm: boolean;
  tensor_count: number;
  input_mib: number;
  output_mib: number;
  ratio: number;
  elapsed_s: number;
}

export function isTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

const FALLBACK_URL = "http://127.0.0.1:11435";

export async function fetchStudioState(): Promise<StudioState> {
  if (isTauriRuntime()) {
    return invoke<StudioState>("get_studio_state");
  }
  const r = await fetch(`${FALLBACK_URL}/v1/vxt/status`);
  return r.json();
}

export async function loadModel(path: string, useGpu = true): Promise<StudioState> {
  if (isTauriRuntime()) {
    return invoke<StudioState>("load_model_cmd", { path, useGpu });
  }
  const r = await fetch(`${FALLBACK_URL}/v1/vxt/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, use_gpu: useGpu }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data?.error?.message || "Failed to load model");
  return data;
}

export async function unloadModel(): Promise<StudioState> {
  if (isTauriRuntime()) {
    return invoke<StudioState>("unload_model_cmd");
  }
  return fetchStudioState();
}

export async function resetChatCache(baseUrl?: string): Promise<void> {
  if (isTauriRuntime()) {
    await invoke("reset_chat_cmd");
    return;
  }
  const root = (baseUrl || `${FALLBACK_URL}/v1`).replace(/\/v1$/, "");
  await fetch(`${root}/v1/vxt/reset`, { method: "POST" });
}

export async function abortGeneration(baseUrl?: string): Promise<void> {
  if (isTauriRuntime()) {
    await invoke("abort_chat_cmd");
    return;
  }
  const root = (baseUrl || `${FALLBACK_URL}/v1`).replace(/\/v1$/, "");
  await fetch(`${root}/v1/vxt/abort`, { method: "POST" });
}

export async function inspectTensorFile(path: string): Promise<InspectReport> {
  return invoke<InspectReport>("inspect_file_cmd", { path });
}

export async function convertToVxt(
  inputPath: string,
  outputPath: string,
  mode: string,
  packAsLlm: boolean,
): Promise<ConvertReport> {
  return invoke<ConvertReport>("convert_vxt_cmd", {
    inputPath,
    outputPath,
    mode,
    packAsLlm,
  });
}

export async function createDemoTensor(): Promise<InspectReport> {
  return invoke<InspectReport>("create_demo_tensor_cmd");
}
