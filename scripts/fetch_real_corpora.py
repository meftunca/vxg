#!/usr/bin/env python3
"""Fetch and verify genuinely real, publicly reproducible benchmark corpora.

None of the files produced by this script use synthetic generators:
1. nyc-taxi-2024-01.csv      — First 300,000 rows of official NYC TLC Yellow Taxi (Jan 2024)
2. nyc-taxi-2024-01.ndjson   — First 180,000 rows of official NYC TLC Yellow Taxi (Jan 2024)
3. gharchive-2025-01-15.ndjson — First 12,000 complete GitHub Archive public API events
4. apache-access.clf         — Elastic's real Apache HTTP Combined Log Format dataset (unmodified)
5. pg100-shakespeare.txt     — Project Gutenberg Complete Works of Shakespeare (unmodified)
6. kodak-truecolor-16.bmp    — 4x4 mosaic of 16 distinct Kodak Lossless True Color camera photos (kodim01..kodim16, 24-bpp BMP)
"""

import gzip
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import urllib.request

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vxg-public-real"

URLS = {
    "nyc_parquet": "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
    "gharchive_gz": "https://data.gharchive.org/2025-01-15-15.json.gz",
    "apache_clf": "https://raw.githubusercontent.com/elastic/examples/master/Common%20Data%20Formats/apache_logs/apache_logs",
    "shakespeare": "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "kodak_fmt": "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "vxg-bench-corpus/0.4.6"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest + ".part", "wb") as out:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            out.write(buf)
    os.replace(dest + ".part", dest)


def fetch_gharchive_lines(url: str, dest: str, max_lines: int = 12000) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    req = urllib.request.Request(url, headers={"User-Agent": "vxg-bench-corpus/0.4.6"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        gz = gzip.GzipFile(fileobj=resp)
        with open(dest + ".part", "wb") as out:
            for i in range(max_lines):
                line = gz.readline()
                if not line:
                    break
                out.write(line)
    os.replace(dest + ".part", dest)


def export_nyc_taxi(parquet_path: str, csv_dest: str, ndjson_dest: str) -> None:
    if os.path.exists(csv_dest) and os.path.exists(ndjson_dest):
        return
    import duckdb

    con = duckdb.connect()
    # Select deterministic integer/float/string/timestamp columns cleanly
    if not os.path.exists(csv_dest):
        con.execute(
            f"""
            COPY (
                SELECT
                    VendorID,
                    strftime(tpep_pickup_datetime, '%Y-%m-%dT%H:%M:%SZ') AS pickup_ts,
                    strftime(tpep_dropoff_datetime, '%Y-%m-%dT%H:%M:%SZ') AS dropoff_ts,
                    COALESCE(passenger_count, 0)::BIGINT AS passenger_count,
                    ROUND(trip_distance * 100)::BIGINT AS distance_centimiles,
                    COALESCE(RatecodeID, 1)::BIGINT AS rate_code,
                    COALESCE(store_and_fwd_flag, 'N') AS store_and_fwd,
                    PULocationID,
                    DOLocationID,
                    payment_type,
                    ROUND(GREATEST(fare_amount, 0) * 100)::BIGINT AS fare_cents,
                    ROUND(GREATEST(tip_amount, 0) * 100)::BIGINT AS tip_cents,
                    ROUND(GREATEST(total_amount, 0) * 100)::BIGINT AS total_cents
                FROM read_parquet('{parquet_path}')
                LIMIT 300000
            ) TO '{csv_dest}.part' (HEADER, DELIMITER ',');
            """
        )
        os.replace(csv_dest + ".part", csv_dest)

    if not os.path.exists(ndjson_dest):
        con.execute(
            f"""
            COPY (
                SELECT
                    VendorID AS vendor_id,
                    strftime(tpep_pickup_datetime, '%Y-%m-%dT%H:%M:%SZ') AS pickup_ts,
                    strftime(tpep_dropoff_datetime, '%Y-%m-%dT%H:%M:%SZ') AS dropoff_ts,
                    COALESCE(passenger_count, 0)::BIGINT AS passenger_count,
                    ROUND(trip_distance * 100)::BIGINT AS distance_centimiles,
                    COALESCE(RatecodeID, 1)::BIGINT AS rate_code,
                    COALESCE(store_and_fwd_flag, 'N') AS store_and_fwd,
                    PULocationID AS pu_loc,
                    DOLocationID AS do_loc,
                    payment_type,
                    ROUND(GREATEST(fare_amount, 0) * 100)::BIGINT AS fare_cents,
                    ROUND(GREATEST(tip_amount, 0) * 100)::BIGINT AS tip_cents,
                    ROUND(GREATEST(total_amount, 0) * 100)::BIGINT AS total_cents
                FROM read_parquet('{parquet_path}')
                LIMIT 200000
            ) TO '{ndjson_dest}.part' (FORMAT JSON);
            """
        )
        os.replace(ndjson_dest + ".part", ndjson_dest)


def read_bmp_24bpp_pixels(bmp_path: str) -> tuple[int, int, bytes]:
    data = open(bmp_path, "rb").read()
    assert data[:2] == b"BM"
    pixel_off = struct.unpack_from("<I", data, 10)[0]
    w, h = struct.unpack_from("<ii", data, 18)
    bpp = struct.unpack_from("<H", data, 28)[0]
    ah = abs(h)
    row_out = w * 3
    if bpp == 24:
        stride = (w * 3 + 3) & ~3
        rows = []
        for r in range(ah):
            src_r = (ah - 1 - r) if h > 0 else r
            off = pixel_off + src_r * stride
            rows.append(data[off : off + row_out])
        return w, ah, b"".join(rows)
    elif bpp == 32:
        stride = w * 4
        rows = []
        for r in range(ah):
            src_r = (ah - 1 - r) if h > 0 else r
            off = pixel_off + src_r * stride
            line = bytearray(row_out)
            raw = data[off : off + stride]
            line[0::3] = raw[0::4]
            line[1::3] = raw[1::4]
            line[2::3] = raw[2::4]
            rows.append(bytes(line))
        return w, ah, b"".join(rows)
    raise ValueError(f"unsupported bpp={bpp}")


def build_kodak_mosaic_bmp(cache_dir: str, dest_bmp: str) -> None:
    if os.path.exists(dest_bmp) and os.path.getsize(dest_bmp) > 0:
        return
    tile_w, tile_h = 768, 512
    grid_x, grid_y = 4, 4
    full_w, full_h = tile_w * grid_x, tile_h * grid_y
    canvas = [bytearray(full_w * 3) for _ in range(full_h)]

    for idx in range(16):
        n = idx + 1
        png_path = os.path.join(cache_dir, f"kodim{n:02d}.png")
        download(URLS["kodak_fmt"].format(n), png_path)
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as tmp:
            tmp_bmp = tmp.name
        try:
            subprocess.run(
                ["sips", "-s", "format", "bmp", png_path, "--out", tmp_bmp],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            w, h, pix = read_bmp_24bpp_pixels(tmp_bmp)
            # Rotate 512x768 portrait Kodak images 90 deg so every cell is 768x512 without scaling
            if (w, h) == (512, 768):
                rot = bytearray(tile_w * tile_h * 3)
                for y in range(h):
                    for x in range(w):
                        src = (y * w + x) * 3
                        dst = (x * tile_w + y) * 3
                        rot[dst : dst + 3] = pix[src : src + 3]
                pix = bytes(rot)
                w, h = tile_w, tile_h
            assert (w, h) == (tile_w, tile_h), f"unexpected size {w}x{h}"
            gx = idx % grid_x
            gy = idx // grid_x
            for y in range(tile_h):
                row_dst = canvas[gy * tile_h + y]
                x0 = gx * tile_w * 3
                row_dst[x0 : x0 + tile_w * 3] = pix[y * tile_w * 3 : (y + 1) * tile_w * 3]
        finally:
            if os.path.exists(tmp_bmp):
                os.remove(tmp_bmp)

    pixel_bytes = b"".join(canvas[::-1])  # standard bottom-up 24-bpp BMP
    file_size = 54 + len(pixel_bytes)
    hdr = struct.pack(
        "<2sIHHIiiiHHIIIIII",
        b"BM",
        file_size,
        0,
        0,
        54,
        40,
        full_w,
        full_h,
        1,
        24,
        0,
        len(pixel_bytes),
        2835,
        2835,
        0,
        0,
    )
    with open(dest_bmp + ".part", "wb") as f:
        f.write(hdr)
        f.write(pixel_bytes)
    os.replace(dest_bmp + ".part", dest_bmp)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    cache_dir = os.path.join(OUT_DIR, "_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("1/5 Fetching NYC TLC Yellow Taxi Jan 2024 Parquet...", flush=True)
    nyc_pq = os.path.join(cache_dir, "yellow_tripdata_2024-01.parquet")
    download(URLS["nyc_parquet"], nyc_pq)
    nyc_csv = os.path.join(OUT_DIR, "nyc-taxi-2024-01.csv")
    nyc_ndjson = os.path.join(OUT_DIR, "nyc-taxi-2024-01.ndjson")
    export_nyc_taxi(nyc_pq, nyc_csv, nyc_ndjson)

    print("2/5 Streaming GitHub Archive 2025-01-15-15 real public events...", flush=True)
    gh_ndjson = os.path.join(OUT_DIR, "gharchive-2025-01-15.ndjson")
    fetch_gharchive_lines(URLS["gharchive_gz"], gh_ndjson, max_lines=12000)

    print("3/5 Fetching Elastic Apache HTTP Combined Log...", flush=True)
    apache_clf = os.path.join(OUT_DIR, "apache-access.clf")
    download(URLS["apache_clf"], apache_clf)

    print("4/5 Fetching Project Gutenberg Shakespeare (pg100.txt)...", flush=True)
    shakes = os.path.join(OUT_DIR, "pg100-shakespeare.txt")
    download(URLS["shakespeare"], shakes)

    print("5/5 Building 16-photo Kodak Lossless True Color 24-bpp BMP...", flush=True)
    kodak_bmp = os.path.join(OUT_DIR, "kodak-truecolor-16.bmp")
    build_kodak_mosaic_bmp(cache_dir, kodak_bmp)

    files = [nyc_csv, nyc_ndjson, gh_ndjson, apache_clf, shakes, kodak_bmp]
    if os.path.exists("/var/log/install.log"):
        inst_copy = os.path.join(OUT_DIR, "macos-install.log")
        if not os.path.exists(inst_copy):
            with open("/var/log/install.log", "rb") as s, open(inst_copy, "wb") as d:
                d.write(s.read())
        files.append(inst_copy)

    manifest = []
    for p in files:
        sz = os.path.getsize(p)
        digest = sha256_file(p)
        manifest.append({"name": os.path.basename(p), "bytes": sz, "sha256": digest})
        print(f"  {os.path.basename(p):28s}  {sz:10,d} B  sha256={digest[:16]}...")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
