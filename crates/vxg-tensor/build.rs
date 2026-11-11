use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn run(cmd: &mut Command, help: &str) {
    let command = format!("{cmd:?}");
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("{help}\n{command}\n{e}"));
    assert!(status.success(), "{help}\n{command}\nexit: {status}");
}

fn archive(out: &Path, name: &str, objects: &[PathBuf]) {
    let mut cmd = Command::new(env::var("AR").unwrap_or_else(|_| "ar".into()));
    cmd.arg("crs").arg(out.join(format!("lib{name}.a")));
    for obj in objects {
        cmd.arg(obj);
    }
    run(&mut cmd, "A native archiver (ar) is required");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(vxt_has_metal)");
    println!("cargo:rerun-if-changed=native/metal_internal.h");
    println!("cargo:rerun-if-changed=native/metal_cb4_fast.h");
    println!("cargo:rerun-if-changed=native/metal_gemv.mm");
    println!("cargo:rerun-if-changed=native/metal_llm.mm");
    println!("cargo:rerun-if-changed=native/metal_llm_turbo.mm");
    println!("cargo:rerun-if-changed=native/metal_cb4_fast.mm");
    println!("cargo:rerun-if-changed=native/metal_cb4_fast_run.mm");
    println!("cargo:rerun-if-changed=native/metal_vision.mm");
    println!("cargo:rerun-if-changed=native/shaders/vxt_gemv.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_llm_ops.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_turbo.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_llm_turbo.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_cb4_fast.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_cb4_fast_ops.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_cb4_fast_gqa.metal");
    println!("cargo:rerun-if-changed=native/shaders/vxt_vision.metal");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-env-changed=VXG_NO_METAL");

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "macos" && env::var_os("VXG_NO_METAL").is_none() {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let gemv_src = fs::read_to_string("native/shaders/vxt_gemv.metal")
            .expect("missing native/shaders/vxt_gemv.metal");
        let llm_src = fs::read_to_string("native/shaders/vxt_llm_ops.metal")
            .expect("missing native/shaders/vxt_llm_ops.metal");
        let turbo_src = fs::read_to_string("native/shaders/vxt_turbo.metal")
            .expect("missing native/shaders/vxt_turbo.metal");
        let llm_turbo_src = fs::read_to_string("native/shaders/vxt_llm_turbo.metal")
            .expect("missing native/shaders/vxt_llm_turbo.metal");
        let cb4_fast_src = fs::read_to_string("native/shaders/vxt_cb4_fast.metal")
            .expect("missing native/shaders/vxt_cb4_fast.metal");
        let cb4_fast_ops_src = fs::read_to_string("native/shaders/vxt_cb4_fast_ops.metal")
            .expect("missing native/shaders/vxt_cb4_fast_ops.metal");
        let cb4_fast_gqa_src = fs::read_to_string("native/shaders/vxt_cb4_fast_gqa.metal")
            .expect("missing native/shaders/vxt_cb4_fast_gqa.metal");
        let vision_src = fs::read_to_string("native/shaders/vxt_vision.metal")
            .expect("missing native/shaders/vxt_vision.metal");
        fs::write(
            out.join("vxt_metal_source.h"),
            format!(
                "#pragma once\nstatic const char* VXT_METAL_SOURCE = R\"VXT_MSL({gemv_src}\n{llm_src}\n{turbo_src}\n{llm_turbo_src}\n{cb4_fast_src}\n{cb4_fast_ops_src}\n{cb4_fast_gqa_src}\n{vision_src})VXT_MSL\";\n"
            ),
        )
        .unwrap();

        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".into());
        let mut objects = Vec::new();
        for (src, obj_name) in [
            ("native/metal_gemv.mm", "metal_gemv.o"),
            ("native/metal_llm.mm", "metal_llm.o"),
            ("native/metal_llm_turbo.mm", "metal_llm_turbo.o"),
            ("native/metal_cb4_fast.mm", "metal_cb4_fast.o"),
            ("native/metal_cb4_fast_run.mm", "metal_cb4_fast_run.o"),
            ("native/metal_vision.mm", "metal_vision.o"),
        ] {
            let obj = out.join(obj_name);
            run(
                Command::new(&cxx)
                    .args([
                        "-std=c++17",
                        "-O3",
                        "-fPIC",
                        "-fobjc-arc",
                        "-Inative",
                        "-c",
                        src,
                    ])
                    .arg(format!("-I{}", out.display()))
                    .arg("-o")
                    .arg(&obj),
                "Xcode Command Line Tools (clang++) required on macOS",
            );
            objects.push(obj);
        }
        archive(&out, "vxt_metal_gemv", &objects);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-cfg=vxt_has_metal");
    }
}
