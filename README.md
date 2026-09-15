# VXG 0.1 — generic lossless compression araştırma prototipi

**Rust CPU çekirdeği + gerçek Metal/CUDA/Vulkan kernel ve host entegrasyon kaynakları.**

Bu bir performans araştırması başlangıcıdır; LZ4 veya Zstandard uyumlu değildir.
"LZ4'ten hızlı", "tüm GPU'larda çalışması doğrulandı" veya "production-ready" iddiası yoktur.

> **Teslim anındaki kritik sınır:** Hazırlama ortamında `rustc`, `cargo`, Metal/CUDA SDK,
> GLSL/SPIR-V derleyicisi ve kullanılabilir GPU yoktu. Dolayısıyla Rust kaynakları ve native
> GPU köprüleri burada derlenmedi. Ortak GPU algoritması C++ SIMT emülatöründe derlenip
> ASan/UBSan ile çalıştırıldı; bağımsız Python modeliyle çapraz doğrulandı. Rust/SDK derleme
> hataları hâlâ mümkün. Tam durum: [docs/VALIDATION.md](docs/VALIDATION.md).

## İçindekiler

| Alan | Bu pakette olan |
|---|---|
| Genel codec | Bağımsız bloklar; RAW, literal, MATCH, REP, FILL, PATTERN2/4/8 |
| Format | 16-bit kontrol sözcüğü; ayrı kontrol, attribute, literal akışları; sabit uzunluk extension |
| CPU | Runtime-dispatch scalar/SSE2/AVX2/AVX-512/NEON; exact-bounds copy ve match extension |
| CRC | CRC32C; x86 SSE4.2 ve ARM CRC dispatch; software fallback |
| Paralellik | Bağımsız bloklar üzerinde scoped Rust threads; blok bazlı random access |
| Metal | MSL encode/decode, Objective-C++ host, paylaşımlı MTLBuffer, kalıcı pipeline/buffer |
| SPIR-V | GLSL compute → build sırasında SPIR-V; gerçek Vulkan compute host |
| CUDA | Encode/decode kernel, CUDA runtime host, stream, event timing, yeniden kullanılan device buffer |
| Test | Rust property/golden tests, fuzz targets, dar kapsamlı Kani harness'leri, bağımsız model |
| Benchmark | Gerçek C liblz4/libzstd ile CPU karşılaştırma; ayrıca GPU host end-to-end profili |

**GPU encoder placeholder değildir**, fakat v0.1'de greedy karar verici blok başına tek lane'de
çalışır. Hash-anchor discovery ve output packing paraleldir. Decode, her blok için 64 lane ile
çalışır. Bu tercih doğrulanabilir bir başlangıç içindir; maksimum GPU kullanımını garanti etmez.

## M2 Max üzerinde ilk adımlar

Rust **1.89 veya sonrası**, Xcode Command Line Tools ve benchmark için Homebrew LZ4/Zstd gerekir.
Core, Cargo bağımlılığı olmadan derlenir; native özellikler ayrı seçilir.

```sh
cd vxg
cargo test --release
cargo run --release --bin vxg -- hardware

# İki native benchmark kütüphanesi; sisteminde yoksa:
brew install lz4 zstd
cargo run --release --features bench --bin vxg-bench -- \
  --threads 1 --size-mib 16 --samples 5 --csv reports/m2-cpu.csv

# Metal gerçek cihaz conformance kontrolü:
cargo run --release --features metal --bin vxg -- gpu-check metal

# CPU + referans kütüphaneler + ayrı GPU e2e satırları:
cargo run --release --features bench,metal --bin vxg-bench -- \
  --gpu metal --size-mib 64 --csv reports/m2-metal.csv
```

GPU context/pipeline oluşturma benchmarkın sıcak döngüsü dışında; transfer, host validation,
framing ve çıktı allocation maliyetleri GPU e2e satırlarının içindedir. Kernel timestamp'i
ayrı, tek gözlem olarak yazılır; CPU e2e hızıyla karıştırılmaz.

## Dosya sıkıştırma / açma

```sh
cargo build --release --bin vxg
./target/release/vxg c input.bin input.vxg --threads 4
./target/release/vxg d input.vxg restored.bin --threads 4
cmp input.bin restored.bin
./target/release/vxg info input.vxg

# Metal örneği; önce --features metal ile derle:
cargo build --release --features metal --bin vxg
./target/release/vxg c input.bin metal.vxg --backend metal
./target/release/vxg d metal.vxg restored-metal.bin --backend metal
```

Varsayılan blok 65,536 byte; checksum açık; CPU tek thread. Dosya var ise üzerine yazılmaz;
`--force` açıkça verilmelidir. CLI bütün dosyayı belleğe alır; streaming API henüz yoktur.
Varsayılan açılmış çıktı limiti 1 GiB'dir; büyük dosyalarda `--max-output BYTES` ayarlanır.

## Linux / SPIR-V / CUDA

```sh
# CPU benchmark için Debian/Ubuntu örneği:
sudo apt-get install build-essential liblz4-dev libzstd-dev
cargo test --release --features bench

# Vulkan/GLSL→SPIR-V için; ayrıca gerçek bir Vulkan ICD/driver gerekir:
sudo apt-get install libvulkan-dev glslang-tools
cargo run --release --features vulkan --bin vxg -- gpu-check vulkan

# CUDA toolkit ve NVIDIA driver kurulu makinede:
export CUDA_HOME=/usr/local/cuda
export VXG_CUDA_ARCH=86  # Örnek: kendi GPU/toolkit hedefine göre değiştir.
cargo run --release --features cuda --bin vxg -- gpu-check cuda
```

`--features vulkan` GLSL kaynağını **gerçek SPIR-V binary'sine derler**; pakette önceden
üretilmiş/denenmiş `.spv` yoktur. Native build köprüleri macOS Metal ve Linux CUDA/Vulkan
hedeflidir; Windows native GPU desteği bu sürümde vaat edilmez. `--all-features` platformlar
arası uygun bir test komutu değildir. Eksik backend sessizce CPU'ya düşmez, hata döndürür.

## Matematiksel kapsam

`N` girdi byte sayısı, `B` blok boyutu iken:

```text
encoded_size <= N + 32 + 32 * ceil(N / B)
CPU hash probes <= N
8 * accepted_matches <= matched_bytes <= N
```

Doğruluk, komut semantiği üzerinden indüksiyonla gösterilir. Ancak bunlar, Rust compiler,
unsafe intrinsics, driver ve shader compiler dahil bütün uygulamanın mekanik ispatı değildir.
Kani harness'leri kontrol sözcüğü ve sınırlı frame-size aritmetiğini hedefler; burada çalıştırılmadı.
Açıklamalar, varsayımlar ve önceki teorik iddiaların düzeltmeleri [PROOFS.md](docs/PROOFS.md)'de.

## Burada gerçekten çalıştırılan kontroller

- Aynı `core.gpu` kaynağı, ASan + UBSan ile WG=8 ve WG=64 emülasyonlarında ayrı ayrı **110** encode/decode vakası.
- Her konfigürasyonda bağımsız model ile **46** yön/vaka çapraz kontrolü; tüm opcode golden'ları dahil.
- **65,536** kontrol sözcüğü, **1,548** sembolik periodicity/doubling vakası, **2,047** küçük ikili girdi,
  **143** sınır/random/periodic turu, **4,700** kesilmiş golden prefix reddi.
- Gerçek kurulu **liblz4 1.10.0 / libzstd 1.5.7**, C++ çağırıcıyla referans-only ölçüldü.
  `reports/reference-c-baselines.csv` **VXG karşılaştırması değildir**; VXG hız satırı içermez.

Bunlar Rust testlerinin geçtiği veya GPU donanım doğrulaması anlamına gelmez.

## Proje haritası

```text
src/                     Rust codec, parser, SIMD, CRC, GPU API
src/bin/                 CLI ve LZ4/Zstd benchmark
native/shaders/core.gpu   Üç GPU diline ortak algoritma
native/                  Metal, Vulkan, CUDA host köprüleri
verification/            Bağımsız Python model ve C++ SIMT emülatörü
tests/vectors/           Modelden üretilmiş geçerli/geçersiz wire golden'ları
fuzz/                    cargo-fuzz hedefleri
src/proofs.rs            Kani proof harness'leri
scripts/                 Test, shader, benchmark ve backend kontrolleri
reports/                 Bu ortamda gerçek çalıştırma raporları
docs/                    Format, ispat, benchmark, güvenlik ve GPU mimarisi
```

[Format](docs/FORMAT.md) · [İspatlar](docs/PROOFS.md) · [Benchmark](docs/BENCHMARKS.md) ·
[GPU](docs/GPU.md) · [Güvenlik](docs/SAFETY.md) · [Validation](docs/VALIDATION.md) ·
[Optimizasyon programı](docs/PERFORMANCE_PLAN.md) · [Kaynaklar](docs/REFERENCES.md)

Lisans: MIT. Araştırma formatı; tek kopyası bulunan veriler üzerinde production arşivleme için kullanma.
