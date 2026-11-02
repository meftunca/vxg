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
    println!("cargo:rustc-check-cfg=cfg(vxt_has_metal)");
    println!("cargo:rerun-if-changed=native/metal_gemv.mm");
    println!("cargo:rerun-if-changed=native/shaders/vxt_gemv.metal");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os == "macos" {
        let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let source = fs::read_to_string("native/shaders/vxt_gemv.metal")
            .expect("missing native/shaders/vxt_gemv.metal");
        fs::write(
            out.join("vxt_metal_source.h"),
            format!("#pragma once\nstatic const char* VXT_METAL_SOURCE = R\"VXT_MSL({source})VXT_MSL\";\n"),
        )
        .unwrap();

        let cxx = env::var("CXX").unwrap_or_else(|_| "clang++".into());
        let object = out.join("metal_gemv.o");
        run(
            Command::new(&cxx)
                .args([
                    "-std=c++17",
                    "-O3",
                    "-fPIC",
                    "-fobjc-arc",
                    "-c",
                    "native/metal_gemv.mm",
                ])
                .arg(format!("-I{}", out.display()))
                .arg("-o")
                .arg(&object),
            "Xcode Command Line Tools (clang++) required on macOS",
        );
        archive(&out, "vxt_metal_gemv", &object);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-cfg=vxt_has_metal");
    }
}
