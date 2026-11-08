//! Explicit diagnostic, create-new outputs only.
use std::{error::Error, fs, io::Write};
use vxg_tensor::{
    llm_pack::{pack_gguf_model_with_activation_profile, turbo_activation_manifest},
    mmap::TensorBytes,
};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 && a.len() != 5 {
        return Err("manifest SOURCE NEW_JSON | pack SOURCE PROFILE NEW_VXT".into());
    }
    let b = TensorBytes::open(&a[2])?;
    let (dest, out) = match a[1].as_str() {
        "manifest" if a.len() == 4 => (
            &a[3],
            serde_json::to_vec_pretty(&turbo_activation_manifest(b.as_slice())?)?,
        ),
        "pack" if a.len() == 5 => {
            let p = serde_json::from_slice(&fs::read(&a[3])?)?;
            (
                &a[4],
                pack_gguf_model_with_activation_profile(b.as_slice(), 2, &p)?.1,
            )
        }
        _ => return Err("invalid diagnostic mode".into()),
    };
    fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dest)?
        .write_all(&out)?;
    println!("{} bytes -> {}", out.len(), dest);
    Ok(())
}
