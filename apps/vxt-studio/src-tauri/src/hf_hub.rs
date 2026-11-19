//! Direct HuggingFace Hub downloader (`curl`-backed with `.part` resume, byte progress, and
//! instant cancellation) plus safe `.vxt` model library file management for `VXT Studio`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::Value;
use vxg_tensor::{Error, Result};

#[derive(Clone, Serialize)]
pub struct ForgeProgressPayload {
    pub phase: String,
    pub current: u64,
    pub total: u64,
    pub percent: f64,
    pub detail: String,
}

#[derive(Clone)]
struct RemoteFile {
    rfilename: String,
    size: u64,
}

fn validate_repo_id(repo_id: &str) -> Result<(&str, &str)> {
    let trimmed = repo_id.trim().trim_matches('/');
    let Some((org, repo)) = trimmed.split_once('/') else {
        return Err(Error::Format(
            "HuggingFace repo kimliği 'org/model' biçiminde olmalı (örn. Qwen/Qwen2.5-1.5B-Instruct)"
                .into(),
        ));
    };
    let valid_part = |s: &str| {
        !s.is_empty()
            && !s.contains('/')
            && !s.contains("..")
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    if !valid_part(org) || !valid_part(repo) {
        return Err(Error::Format(format!(
            "Geçersiz HuggingFace repo kimliği: '{trimmed}'"
        )));
    }
    Ok((org, repo))
}

pub fn default_hf_vxt_output_path(repo_id: &str, mode: &str) -> Result<PathBuf> {
    let (_org, repo) = validate_repo_id(repo_id)?;
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let models_dir = Path::new(&home).join("Models");
    fs::create_dir_all(&models_dir).map_err(|e| Error::Io(e.to_string()))?;
    let clean_mode = mode.trim().to_ascii_lowercase();
    Ok(models_dir.join(format!("{repo}-{clean_mode}.vxt")))
}

fn fetch_repo_manifest(repo_id: &str, hf_token: Option<&str>) -> Result<Vec<RemoteFile>> {
    let api_url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");
    let mut cmd = Command::new("/usr/bin/curl");
    cmd.args(["-sSL", "--fail-with-body", "--max-time", "30"]);
    if let Some(tok) = hf_token.map(str::trim).filter(|t| !t.is_empty()) {
        cmd.arg("-H").arg(format!("Authorization: Bearer {tok}"));
    }
    cmd.arg(&api_url);
    let out = cmd
        .output()
        .map_err(|e| Error::Io(format!("curl çalıştırılamadı: {e}")))?;
    if !out.status.success() {
        let body = String::from_utf8_lossy(&out.stderr);
        let msg = String::from_utf8_lossy(&out.stdout);
        return Err(Error::Io(format!(
            "HuggingFace API hatası ({repo_id}): {} {}",
            body.trim(),
            msg.chars().take(180).collect::<String>()
        )));
    }
    let root: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| Error::Format(format!("HF manifest JSON okunamadı: {e}")))?;
    let Some(siblings) = root.get("siblings").and_then(Value::as_array) else {
        return Err(Error::Format("HF manifest içinde 'siblings' bulunamadı".into()));
    };
    let mut files = Vec::with_capacity(siblings.len());
    for item in siblings {
        let Some(name) = item.get("rfilename").and_then(Value::as_str) else {
            continue;
        };
        let size = item
            .get("size")
            .and_then(Value::as_u64)
            .or_else(|| item.get("lfs").and_then(|l| l.get("size")).and_then(Value::as_u64))
            .unwrap_or(0);
        files.push(RemoteFile {
            rfilename: name.to_owned(),
            size,
        });
    }
    Ok(files)
}

fn select_download_plan(files: &[RemoteFile]) -> Result<(Vec<RemoteFile>, bool)> {
    let top_level: Vec<&RemoteFile> = files
        .iter()
        .filter(|f| !f.rfilename.contains('/'))
        .collect();
    let has_cfg = top_level.iter().any(|f| f.rfilename == "config.json");
    let has_st = top_level
        .iter()
        .any(|f| f.rfilename.ends_with(".safetensors"));

    if has_cfg && has_st {
        let has_non_consolidated = top_level.iter().any(|f| {
            f.rfilename.ends_with(".safetensors") && !f.rfilename.starts_with("consolidated")
        });
        let meta_names = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "preprocessor_config.json",
            "chat_template.jinja",
            "model.safetensors.index.json",
        ];
        let mut plan = Vec::new();
        for f in &top_level {
            if meta_names.contains(&f.rfilename.as_str()) {
                plan.push((*f).clone());
            } else if f.rfilename.ends_with(".safetensors") {
                if has_non_consolidated && f.rfilename.starts_with("consolidated") {
                    continue;
                }
                plan.push((*f).clone());
            }
        }
        return Ok((plan, false));
    }

    let mut ggufs: Vec<&RemoteFile> = top_level
        .into_iter()
        .filter(|f| f.rfilename.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if !ggufs.is_empty() {
        ggufs.sort_by_key(|f| {
            let l = f.rfilename.to_ascii_lowercase();
            if l.contains("bf16") || l.contains("f16") {
                0u8
            } else if l.contains("q8_0") {
                1
            } else if l.contains("q4_k_m") {
                2
            } else {
                3
            }
        });
        return Ok((vec![ggufs[0].clone()], true));
    }

    Err(Error::Format(
        "Bu HuggingFace reposunda desteklenen 'config.json + .safetensors' veya '.gguf' bulunamadı"
            .into(),
    ))
}

/// Downloads a HuggingFace repository into `~/Models/hf-hub/<org>--<repo>/` and returns the path
/// to pass into `convert_tensor_to_vxt_with_progress` (either the HF directory or a `.gguf` file).
pub fn download_hf_repo(
    repo_id: &str,
    hf_token: Option<&str>,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(ForgeProgressPayload) + Sync),
) -> Result<PathBuf> {
    let (org, repo) = validate_repo_id(repo_id)?;
    let clean_repo = format!("{org}/{repo}");
    on_progress(ForgeProgressPayload {
        phase: "download".into(),
        current: 0,
        total: 100,
        percent: 1.0,
        detail: format!("{clean_repo} manifest bilgisi alınıyor..."),
    });

    let all_files = fetch_repo_manifest(&clean_repo, hf_token)?;
    let (plan, is_single_gguf) = select_download_plan(&all_files)?;

    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let dest_dir = Path::new(&home)
        .join("Models")
        .join("hf-hub")
        .join(format!("{org}--{repo}"));
    fs::create_dir_all(&dest_dir).map_err(|e| Error::Io(e.to_string()))?;

    let total_bytes: u64 = plan.iter().map(|f| f.size).sum();
    let mut completed_bytes: u64 = 0;

    for (idx, rf) in plan.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Io("İndirme kullanıcı tarafından iptal edildi".into()));
        }
        let final_path = dest_dir.join(&rf.rfilename);
        if let Ok(meta) = fs::metadata(&final_path) {
            if (rf.size > 0 && meta.len() == rf.size) || (rf.size == 0 && meta.len() > 0) {
                completed_bytes = completed_bytes.saturating_add(meta.len());
                let pct = if total_bytes > 0 {
                    (completed_bytes as f64 / total_bytes as f64) * 100.0
                } else {
                    ((idx + 1) as f64 / plan.len() as f64) * 100.0
                };
                on_progress(ForgeProgressPayload {
                    phase: "download".into(),
                    current: completed_bytes,
                    total: total_bytes,
                    percent: pct.clamp(0.0, 100.0),
                    detail: format!("Önbellekte hazır: {}", rf.rfilename),
                });
                continue;
            }
        }

        let part_path = dest_dir.join(format!("{}.part", rf.rfilename));
        let url = format!("https://huggingface.co/{clean_repo}/resolve/main/{}", rf.rfilename);
        let mut cmd = Command::new("/usr/bin/curl");
        cmd.args(["-L", "-sS", "--fail", "-C", "-"])
            .arg("-o")
            .arg(&part_path);
        if let Some(tok) = hf_token.map(str::trim).filter(|t| !t.is_empty()) {
            cmd.arg("-H").arg(format!("Authorization: Bearer {tok}"));
        }
        cmd.arg(&url)
            .stdout(Stdio::null())
            .stderr(Stdio::piped());

        let mut child = cmd
            .spawn()
            .map_err(|e| Error::Io(format!("curl başlatılamadı: {e}")))?;

        let mut last_emit = Instant::now();
        let status = loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                return Err(Error::Io("İndirme kullanıcı tarafından iptal edildi".into()));
            }
            if let Some(st) = child.try_wait().map_err(|e| Error::Io(e.to_string()))? {
                break st;
            }
            if last_emit.elapsed() >= Duration::from_millis(120) {
                let part_len = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
                let cur = completed_bytes.saturating_add(part_len);
                let pct = if total_bytes > 0 {
                    (cur as f64 / total_bytes as f64) * 100.0
                } else {
                    (idx as f64 / plan.len().max(1) as f64) * 100.0
                };
                let mib_cur = cur as f64 / (1024.0 * 1024.0);
                let mib_tot = total_bytes as f64 / (1024.0 * 1024.0);
                on_progress(ForgeProgressPayload {
                    phase: "download".into(),
                    current: cur,
                    total: total_bytes,
                    percent: pct.clamp(0.0, 99.9),
                    detail: if total_bytes > 0 {
                        format!(
                            "İndiriliyor ({}/{}) {}: {:.1} / {:.1} MiB",
                            idx + 1,
                            plan.len(),
                            rf.rfilename,
                            mib_cur,
                            mib_tot
                        )
                    } else {
                        format!("İndiriliyor ({}/{}) {}", idx + 1, plan.len(), rf.rfilename)
                    },
                });
                last_emit = Instant::now();
            }
            std::thread::sleep(Duration::from_millis(40));
        };

        if !status.success() {
            return Err(Error::Io(format!(
                "'{}' indirilemedi (yetki gerekiyorsa HF Token girin)",
                rf.rfilename
            )));
        }
        let final_len = fs::metadata(&part_path).map(|m| m.len()).unwrap_or(rf.size);
        fs::rename(&part_path, &final_path).map_err(|e| Error::Io(e.to_string()))?;
        completed_bytes = completed_bytes.saturating_add(final_len);
    }

    if is_single_gguf {
        Ok(dest_dir.join(&plan[0].rfilename))
    } else {
        Ok(dest_dir)
    }
}

pub fn delete_vxt_model_file(path: &str) -> Result<()> {
    let p = Path::new(path);
    let is_vxt = p
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.eq_ignore_ascii_case("vxt"))
        .unwrap_or(false);
    if !is_vxt {
        return Err(Error::Format(
            "Güvenlik gereği yalnızca '.vxt' uzantılı model arşivleri silinebilir".into(),
        ));
    }
    if !p.is_file() {
        return Err(Error::Io(format!("Dosya bulunamadı: {path}")));
    }
    fs::remove_file(p).map_err(|e| Error::Io(format!("Dosya silinemedi: {e}")))
}
