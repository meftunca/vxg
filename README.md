# VXG 0.4 — Birleşik Adaptif Kayıpsız Blok Codec'i (Format v4)

**Rust CPU (SIMD/SWAR) çekirdeği + yapısal sütun (NDJSON/CSV/TSV/Logfmt), 2D/Stride görüntü (BMP) ve Fast-LZ+Huffman adaptif blok motoru + native Metal/CUDA/Vulkan kaynakları.**

VXG `v4`, her 64 KiB bağımsız blok için verinin yapısını çalışma zamanında tespit ederek en uygun sıkıştırma motorunu seçen (`BLOCK_COLUMN`, `BLOCK_LZ`, `BLOCK_RAW`), $O(1)$ blok bazlı rastgele erişim (random-access) ve çok çekirdekli paralel sıkıştırma/açma sunan kayıpsız bir codec'tir.

## İçindekiler

| Alan | Bu pakette olan |
|---|---|
| Format v4 | 8-bayt kompakt blok dizini (`payload_size \| kind << 30`, `crc32c`); `v1`–`v3` geriye dönük okuma uyumu |
| Yapısal Sütun Motoru | NDJSON, CSV, TSV ve Logfmt için otomatik şema/ayraç keşfi; `Dictionary`, `DecimalDelta`, `DecimalSuffix`, `Timestamp` (SWAR+NEON), `EpochCounter` (NEON hex), `SegmentDictionary`, `FrontCodedDict` |
| Görüntü / Stride Motoru | 24-bpp / 32-bpp BMP ve çoklu stride (`1, 2, 3, 4, 8`) dikey/yatay delta + `R-G`/`B-G` renk ayrıştırma + 4-akışlı kanonik Huffman |
| Metin & Kaynak Kodu | Sabit-genel (const-generic) Fast-LZ eşleştirici + 4-akışlı kanonik Huffman kodlayıcı/çözücü |
| CPU SIMD & SWAR | Runtime-dispatch scalar/SSE2/AVX2/AVX-512/NEON; 32-bayt dallanmasız NEON dizgi tarama, 8-yollu ağaç indirgemeli bit paketleme |
| CRC & Güvenlik | Donanım hızlandırmalı CRC32C (SSE4.2 / ARMv8 CRC); bozuk/kesilmiş girdilere karşı sınırlandırılmış bellek ve deterministik red |
| Paralellik & Erişim | Bağımsız bloklar üzerinde scoped Rust thread havuzu (`--threads auto`); `vxg block` ile $O(1)$ tekil blok çıkarma |
| Benchmark & CI | Gerçek C `liblz4` / `libzstd` ile 12 veri setinde karşılaştırma (`--suite v4 --markdown`); Linux, macOS ve Windows CI iş akışı |

---

## 12 Veri Seti Performans Karşılaştırması (Tek Çekirdek, Apple Silicon NEON, 64 KiB Blok)

| Veri Seti | Boyut | VXG v4 Oran | LZ4 Oran | Zstd-1 Oran | VXG v4 Encode | Zstd-1 Encode | VXG v4 Decode | LZ4 Decode | Zstd-1 Decode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`telemetry-16MiB.csv`** | 16 MiB | **0.0407** | 0.1994 | 0.0841 | **0.892 GB/s** | 0.819 GB/s | **5.851 GB/s** | 5.215 GB/s | 1.573 GB/s |
| **`events-16MiB.tsv`** | 16 MiB | **0.0334** | 0.2223 | 0.0741 | **1.028 GB/s** | 0.923 GB/s | **5.725 GB/s** | 4.015 GB/s | 1.831 GB/s |
| **`app-logs-16MiB.logfmt`** | 16 MiB | **0.0353** | 0.2007 | 0.0695 | **1.162 GB/s** | 0.933 GB/s | **7.680 GB/s** | 5.008 GB/s | 1.782 GB/s |
| **`json-records-16MiB.ndjson`** | 16 MiB | **0.0226** | 0.1497 | 0.0542 | **1.507 GB/s** | 1.184 GB/s | **8.834 GB/s** | 6.510 GB/s | 2.361 GB/s |
| **`field-order-optional-escaping.ndjson`** | 16 MiB | **0.0445** | 0.2132 | 0.1043 | **0.817 GB/s** | 0.680 GB/s | **4.344 GB/s** | 3.378 GB/s | 1.320 GB/s |
| **`event-mix-identifiers-16MiB.ndjson`** | 16 MiB | **0.0248** | 0.1545 | 0.0779 | **1.085 GB/s** | 0.900 GB/s | **6.981 GB/s** | 4.000 GB/s | 1.632 GB/s |
| **`bursty-variable-records-16MiB.ndjson`** | 16 MiB | **0.0480** | 0.2068 | 0.1197 | **0.675 GB/s** | 0.605 GB/s | **3.144 GB/s** | 3.036 GB/s | 1.213 GB/s |
| **`source-and-docs-16MiB.txt`** | 16 MiB | **0.2908** | 0.4163 | 0.2969 | **0.217 GB/s** | 0.346 GB/s | **1.154 GB/s** | 2.359 GB/s | 0.948 GB/s |
| **`ui-render-32bpp-16MiB.bmp`** | 16 MiB | **0.0041** | 0.0140 | 0.0091 | **6.277 GB/s** | 4.568 GB/s | **10.089 GB/s** | 13.813 GB/s | 3.330 GB/s |
| **`rgb-frame-24bpp-16MiB.bmp`** | 16 MiB | **0.0011** | 0.0227 | 0.0119 | **6.342 GB/s** | 4.989 GB/s | **10.359 GB/s** | 7.355 GB/s | 4.915 GB/s |
| **`knowledge_base_raw.bmp`** | 820 MB | **0.0393** | 0.0920 | 0.0467 | **1.293 GB/s** | 1.643 GB/s | **3.407 GB/s** | 5.734 GB/s | 2.285 GB/s |
| **`surveys_raw.bmp`** | 363 MB | **0.0798** | 0.1075 | 0.0825 | **0.942 GB/s** | 1.294 GB/s | **2.830 GB/s** | 4.319 GB/s | 1.991 GB/s |

---

## Hızlı Başlangıç ve Benchmark

Rust **1.89 veya sonrası** ve benchmark için `liblz4` / `libzstd` gerekir.

```sh
cd vxg
cargo test --release
cargo run --release --bin vxg -- hardware

# Gömülü 10-veri setli v4 çoklu format benchmark paketi (Markdown + CSV çıktılı):
cargo run --release --features bench --bin vxg-bench -- \
  --suite v4 --size-mib 16 --samples 3 --seconds 0.05 \
  --csv reports/local-v4.csv --markdown reports/local-v4.md
```

## CLI Kullanımı (`vxg`)

```sh
cargo build --release --bin vxg

# Dosya sıkıştırma ve açma (--threads auto tüm çekirdekleri kullanır):
./target/release/vxg compress input.ndjson input.vxg --threads auto
./target/release/vxg decompress input.vxg restored.ndjson --threads auto --force

# Çıktı yazmadan frame ve CRC32C bütünlük doğrulaması:
./target/release/vxg verify input.vxg --threads auto

# Frame üstbilgisi, sıkıştırma oranı ve blok türleri (RAW / LZ / COLUMN):
./target/release/vxg info input.vxg --summary

# O(1) rastgele erişim ile tek bir bağımsız bloğu çıkarma:
./target/release/vxg block input.vxg 0 block0.bin --force

# Unix pipeline (stdin / stdout '-') kullanımı:
cat input.ndjson | ./target/release/vxg c - - -q | ./target/release/vxg d - - -q > restored.ndjson
```

Varsayılan blok 65,536 byte; checksum açık; dosya var ise `--force` (`-f`) verilmedikçe üzerine yazılmaz. Varsayılan açılmış çıktı limiti 1 GiB'dir; daha büyük dosyalarda `--max-output BYTES` ayarlanır.

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
