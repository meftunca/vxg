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
fn exists(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
fn archive(out: &Path, name: &str, object: &Path) {
    run(
        Command::new(env::var("AR").unwrap_or_else(|_| "ar".into()))
            .arg("crs")
            .arg(out.join(format!("lib{name}.a")))
            .arg(object),
        "A native archiver (ar) is required",
    );
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
}
fn main() {
    println!("cargo:rustc-check-cfg=cfg(kani)");
    println!("cargo:rustc-check-cfg=cfg(miri)");
    println!("cargo:rerun-if-changed=native");
    for key in [
        "CXX",
        "AR",
        "CUDA_HOME",
        "CUDA_PATH",
        "VULKAN_SDK",
        "GLSLC",
        "VXG_CUDA_ARCH",
        "VXG_LIBRARY_PATH",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let cxx = env::var("CXX").unwrap_or_else(|_| {
        if os == "macos" {
            "clang++".into()
        } else {
            "c++".into()
        }
    });
    if let Some(paths) = env::var_os("VXG_LIBRARY_PATH") {
        for p in env::split_paths(&paths) {
            println!("cargo:rustc-link-search=native={}", p.display());
        }
    }
    if os == "macos" {
        for p in ["/opt/homebrew/lib", "/usr/local/lib"] {
            if Path::new(p).exists() {
                println!("cargo:rustc-link-search=native={p}");
            }
        }
    }
    if env::var_os("CARGO_FEATURE_BENCH").is_some() {
        println!("cargo:rustc-link-lib=lz4");
        println!("cargo:rustc-link-lib=zstd");
    }
    if env::var_os("CARGO_FEATURE_METAL").is_some() {
        assert_eq!(
            os, "macos",
            "The metal feature requires macOS and Xcode Command Line Tools"
        );
        let source = fs::read_to_string("native/shaders/vxg.metal").unwrap();
        fs::write(out.join("vxg_metal_source.h"),format!("#pragma once\nstatic const char* VXG_METAL_SOURCE=R\"VXG_MSL({source})VXG_MSL\";\n")).unwrap();
        let object = out.join("metal.o");
        run(
            Command::new(&cxx)
                .args([
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                    "-fobjc-arc",
                    "-c",
                    "native/metal.mm",
                    "-Inative",
                ])
                .arg(format!("-I{}", out.display()))
                .arg("-o")
                .arg(&object),
            "Install Xcode Command Line Tools: xcode-select --install",
        );
        archive(&out, "vxg_metal", &object);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
    }
    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        assert_eq!(
            os, "linux",
            "The v0.1 Vulkan native bridge targets Linux; use Metal on macOS"
        );
        let sdk = env::var_os("VULKAN_SDK").map(PathBuf::from);
        let glslc = env::var("GLSLC")
            .ok()
            .or_else(|| {
                sdk.as_ref()
                    .map(|p| p.join("bin/glslc").display().to_string())
            })
            .unwrap_or_else(|| "glslc".into());
        let spirv = out.join("vxg.spv");
        if exists(&glslc) {
            run(
                Command::new(&glslc)
                    .args([
                        "--target-env=vulkan1.1",
                        "-fshader-stage=compute",
                        "-O",
                        "native/shaders/vxg.comp",
                        "-o",
                    ])
                    .arg(&spirv),
                "GLSL to SPIR-V compilation failed",
            );
        } else {
            run(
                Command::new("glslangValidator")
                    .args([
                        "-V",
                        "--target-env",
                        "vulkan1.1",
                        "-S",
                        "comp",
                        "native/shaders/vxg.comp",
                        "-o",
                    ])
                    .arg(&spirv),
                "Install Vulkan SDK/glslc or glslangValidator (SPIR-V compiler)",
            );
        }
        let bytes = fs::read(&spirv).unwrap();
        assert!(
            bytes.len() % 4 == 0 && bytes.len() >= 20,
            "invalid SPIR-V size"
        );
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(words[0], 0x07230203);
        let text = words
            .iter()
            .map(|x| format!("0x{x:08x}u"))
            .collect::<Vec<_>>()
            .join(",");
        fs::write(out.join("vxg_spirv.h"),format!("#pragma once\n#include <cstdint>\nstatic const uint32_t VXG_SPIRV[]={{ {text} }};\n")).unwrap();
        let object = out.join("vulkan.o");
        let mut cmd = Command::new(&cxx);
        cmd.args([
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-c",
            "native/vulkan.cpp",
            "-Inative",
        ])
        .arg(format!("-I{}", out.display()))
        .arg("-o")
        .arg(&object);
        if let Some(sdk) = sdk {
            cmd.arg(format!("-I{}", sdk.join("include").display()));
            println!(
                "cargo:rustc-link-search=native={}",
                sdk.join("lib").display()
            );
        }
        run(
            &mut cmd,
            "Install Vulkan headers and the loader library (e.g. libvulkan-dev)",
        );
        archive(&out, "vxg_vulkan", &object);
        println!("cargo:rustc-link-lib=vulkan");
        println!(
            "cargo:rustc-link-lib={}",
            if os == "macos" { "c++" } else { "stdc++" }
        );
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        assert_eq!(
            os, "linux",
            "The v0.1 CUDA native bridge targets Linux with an NVIDIA CUDA toolkit"
        );
        let home = env::var_os("CUDA_HOME")
            .or_else(|| env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
        let full = home.join("bin/nvcc");
        let nvcc = if full.exists() {
            full
        } else {
            PathBuf::from("nvcc")
        };
        let arch = env::var("VXG_CUDA_ARCH").unwrap_or_else(|_| "75".into());
        assert!(
            arch.len() >= 2 && arch.chars().all(|c| c.is_ascii_digit()),
            "VXG_CUDA_ARCH must be numeric, e.g. 75, 86, 89 or 90"
        );
        let object = out.join("cuda.o");
        run(Command::new(nvcc).args(["-std=c++17","-O3","-lineinfo","-Xcompiler=-fPIC","-c","native/cuda.cu","-Inative"])
            .arg(format!("-gencode=arch=compute_{arch},code=[sm_{arch},compute_{arch}]")).arg("-o").arg(&object),"Install CUDA toolkit and set CUDA_HOME; GPU architecture/toolkit compatibility is required");
        archive(&out, "vxg_cuda", &object);
        println!(
            "cargo:rustc-link-search=native={}",
            home.join("lib64").display()
        );
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=stdc++");
    }
}
