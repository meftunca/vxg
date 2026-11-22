//! Zero-dependency HTTP/1.1 Object Store origin (`vxg-kv origin`) supporting `HEAD`,
//! `GET` (including `Range: bytes=s-e` -> `206 Partial Content`), and atomic CAS `PUT`
//! (`If-None-Match: *` and `If-Match: <etag>` -> `412 Precondition Failed`).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;

use crate::store::{LocalFsStore, ObjectStore, PutOutcome};

/// Bind `127.0.0.1:<port>` and serve `dir` as a CAS + Range HTTP/1.1 Object Store.
pub fn serve_origin(dir: &Path, port: u16) -> Result<(), String> {
    let listener =
        TcpListener::bind(("127.0.0.1", port)).map_err(|e| format!("bind 127.0.0.1:{port}: {e}"))?;
    println!(
        "vxg-kv HTTP/1.1 CAS+Range Origin listening on http://127.0.0.1:{port} (dir={})",
        dir.display()
    );
    serve_origin_listener(listener, dir)
}

/// Accept loop over a pre-bound `TcpListener` (supports ephemeral port `0` in tests).
pub fn serve_origin_listener(listener: TcpListener, dir: &Path) -> Result<(), String> {
    let store = Arc::new(LocalFsStore::new(dir)?);
    let active = Arc::new(AtomicUsize::new(0));
    for mut stream in listener.incoming().flatten() {
        if active.load(Ordering::Relaxed) >= 128 {
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            continue;
        }
        active.fetch_add(1, Ordering::AcqRel);
        let st = Arc::clone(&store);
        let act = Arc::clone(&active);
        std::thread::spawn(move || {
            handle_conn(stream, &st);
            act.fetch_sub(1, Ordering::AcqRel);
        });
    }
    Ok(())
}

fn handle_conn(mut stream: TcpStream, store: &LocalFsStore) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    loop {
        let mut req_line = String::new();
        match reader.read_line(&mut req_line) {
            Ok(0) | Err(_) => break,
            Ok(_) if req_line.len() > 4096 => break,
            Ok(_) => {}
        }
        let parts: Vec<&str> = req_line.split_whitespace().collect();
        if parts.len() < 2 {
            break;
        }
        let (method, raw_target) = (parts[0], parts[1]);
        let key = raw_target
            .split_once('?')
            .map_or(raw_target, |(p, _)| p)
            .trim_start_matches('/');

        let mut content_len: usize = 0;
        let mut range_hdr: Option<String> = None;
        let mut if_match: Option<String> = None;
        let mut if_none_match: Option<String> = None;
        let mut keep_alive = true;
        for _ in 0..64 {
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = line.split_once(':') {
                let kt = k.trim();
                let vt = v.trim();
                if kt.eq_ignore_ascii_case("content-length") {
                    content_len = vt.parse().unwrap_or(0);
                } else if kt.eq_ignore_ascii_case("range") {
                    range_hdr = Some(vt.to_owned());
                } else if kt.eq_ignore_ascii_case("if-match") {
                    if_match = Some(vt.to_owned());
                } else if kt.eq_ignore_ascii_case("if-none-match") {
                    if_none_match = Some(vt.to_owned());
                } else if kt.eq_ignore_ascii_case("connection")
                    && vt.eq_ignore_ascii_case("close")
                {
                    keep_alive = false;
                }
            }
        }
        let mut body = vec![0u8; content_len];
        if content_len > 0 && reader.read_exact(&mut body).is_err() {
            break;
        }
        let conn = if keep_alive { "keep-alive" } else { "close" };
        if key.is_empty() || key.contains("..") {
            let _ = write_simple(&mut stream, 400, "Bad Request", conn, None, &[]);
            break;
        }
        let ok = match method {
            "HEAD" => match store.head(key) {
                Ok(Some(meta)) => {
                    let hdr = format!(
                        "HTTP/1.1 200 OK\r\nAccept-Ranges: bytes\r\nETag: {}\r\nContent-Length: {}\r\nConnection: {conn}\r\n\r\n",
                        meta.etag, meta.len
                    );
                    stream.write_all(hdr.as_bytes()).is_ok()
                }
                Ok(None) => write_simple(&mut stream, 404, "Not Found", conn, None, &[]),
                Err(e) => write_simple(&mut stream, 500, "Error", conn, None, e.as_bytes()),
            },
            "GET" => handle_get(&mut stream, store, key, range_hdr.as_deref(), conn),
            "PUT" => handle_put(
                &mut stream,
                store,
                key,
                &body,
                if_match.as_deref(),
                if_none_match.as_deref(),
                conn,
            ),
            _ => write_simple(&mut stream, 405, "Method Not Allowed", conn, None, &[]),
        };
        if !ok || !keep_alive {
            break;
        }
    }
}

fn handle_get(
    stream: &mut TcpStream,
    store: &LocalFsStore,
    key: &str,
    range_hdr: Option<&str>,
    conn: &str,
) -> bool {
    if let Some(r) = range_hdr.and_then(|s| s.strip_prefix("bytes=")) {
        if let Some((a, b)) = r.split_once('-') {
            if let (Ok(s), Ok(e)) = (a.parse::<u64>(), b.parse::<u64>()) {
                match store.head(key) {
                    Ok(Some(meta)) if s <= e && e < meta.len => {
                        match store.get_range(key, s, e + 1) {
                            Ok(chunk) => {
                                let hdr = format!(
                                    "HTTP/1.1 206 Partial Content\r\nAccept-Ranges: bytes\r\nETag: {}\r\nContent-Range: bytes {s}-{e}/{}\r\nContent-Length: {}\r\nConnection: {conn}\r\n\r\n",
                                    meta.etag,
                                    meta.len,
                                    chunk.len()
                                );
                                return stream.write_all(hdr.as_bytes()).is_ok()
                                    && stream.write_all(&chunk).is_ok();
                            }
                            Err(err) => {
                                return write_simple(
                                    stream,
                                    500,
                                    "Error",
                                    conn,
                                    None,
                                    err.as_bytes(),
                                );
                            }
                        }
                    }
                    Ok(None) => return write_simple(stream, 404, "Not Found", conn, None, &[]),
                    Ok(_) => {
                        return write_simple(
                            stream,
                            416,
                            "Range Not Satisfiable",
                            conn,
                            None,
                            &[],
                        )
                    }
                    Err(err) => {
                        return write_simple(stream, 500, "Error", conn, None, err.as_bytes())
                    }
                }
            }
        }
    }
    match store.get(key) {
        Ok(Some((data, etag))) => {
            write_simple(stream, 200, "OK", conn, Some(&etag), &data)
        }
        Ok(None) => write_simple(stream, 404, "Not Found", conn, None, &[]),
        Err(err) => write_simple(stream, 500, "Error", conn, None, err.as_bytes()),
    }
}

fn handle_put(
    stream: &mut TcpStream,
    store: &LocalFsStore,
    key: &str,
    body: &[u8],
    if_match: Option<&str>,
    if_none_match: Option<&str>,
    conn: &str,
) -> bool {
    let outcome = if if_none_match == Some("*") {
        store.put_if_none_match(key, body)
    } else if let Some(expected) = if_match {
        store.put_if_match(key, expected, body)
    } else {
        store.put(key, body).map(PutOutcome::Written)
    };
    match outcome {
        Ok(PutOutcome::Written(etag)) => {
            write_simple(stream, 200, "OK", conn, Some(&etag), &[])
        }
        Ok(PutOutcome::PreconditionFailed) => {
            write_simple(stream, 412, "Precondition Failed", conn, None, &[])
        }
        Err(err) => write_simple(stream, 500, "Error", conn, None, err.as_bytes()),
    }
}

fn write_simple(
    stream: &mut TcpStream,
    code: u16,
    reason: &str,
    conn: &str,
    etag: Option<&str>,
    body: &[u8],
) -> bool {
    let mut hdr = format!("HTTP/1.1 {code} {reason}\r\nAccept-Ranges: bytes\r\n");
    if let Some(et) = etag {
        hdr.push_str(&format!("ETag: {et}\r\n"));
    }
    hdr.push_str(&format!(
        "Content-Length: {}\r\nConnection: {conn}\r\n\r\n",
        body.len()
    ));
    stream.write_all(hdr.as_bytes()).is_ok() && (body.is_empty() || stream.write_all(body).is_ok())
}
