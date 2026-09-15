# Primary technical references

These explain related formats, API contracts and verification tools. They do not prove
VXG performance or certify this implementation. No code from these projects is vendored.
Reference LZ4/Zstd are linked as independently installed libraries in the benchmark feature.

- LZ4 block specification: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
- LZ4 reference API: https://github.com/lz4/lz4/blob/dev/lib/lz4.h
- Zstandard format: https://github.com/facebook/zstd/blob/dev/doc/zstd_compression_format.md
- Zstandard reference API: https://github.com/facebook/zstd/blob/dev/lib/zstd.h
- Rust architecture intrinsics: https://doc.rust-lang.org/core/arch/index.html
- Rust 1.89 AVX-512 stabilization: https://blog.rust-lang.org/2025/08/07/Rust-1.89.0/
- AVX-512 byte comparison: https://doc.rust-lang.org/core/arch/x86_64/fn._mm512_cmpeq_epi8_mask.html
- ARM CRC32C intrinsic: https://doc.rust-lang.org/core/arch/aarch64/fn.__crc32cd.html
- Vulkan pipeline memory barriers: https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html
- Kani installation: https://model-checking.github.io/kani/install-guide.html
- Kani usage: https://model-checking.github.io/kani/usage.html

SDK/driver support and tool versions are environment-specific. Record actual versions
when executing the supplied sources; do not treat this list as an SDK validation report.
