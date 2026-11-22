//! Zero-dependency Redis Geospatial (`GEOADD`, `GEOPOS`, `GEODIST`, `GEOSEARCH`)
//! backed by 52-bit interleaved Geohash scores stored natively inside `Value::ZSet`.

use std::collections::BTreeMap;

use crate::dispatch_ds::{arg_str, parse_f64, parse_i64, WRONGTYPE};
use crate::engine::KvEngine;
use crate::resp::RespVal;
use crate::types::Value;

const GEO_LAT_MIN: f64 = -85.051_128_78;
const GEO_LAT_MAX: f64 = 85.051_128_78;
const GEO_LON_MIN: f64 = -180.0;
const GEO_LON_MAX: f64 = 180.0;
const EARTH_RADIUS_M: f64 = 6_372_797.560_856;

/// Dispatch `GEO*` commands. Returns `None` if command belongs elsewhere.
pub fn dispatch_geo_cmd(eng: &mut KvEngine, cmd: &str, args: &[Vec<u8>]) -> Option<RespVal> {
    Some(match cmd {
        "GEOADD" => dispatch_geoadd(eng, args),
        "GEOPOS" => dispatch_geopos(eng, args),
        "GEODIST" => dispatch_geodist(eng, args),
        "GEOSEARCH" => dispatch_geosearch(eng, args),
        _ => return None,
    })
}

fn geohash_encode(lon: f64, lat: f64) -> Option<u64> {
    if !(GEO_LON_MIN..=GEO_LON_MAX).contains(&lon) || !(GEO_LAT_MIN..=GEO_LAT_MAX).contains(&lat) {
        return None;
    }
    let lon_n = ((lon - GEO_LON_MIN) / (GEO_LON_MAX - GEO_LON_MIN) * ((1u64 << 26) as f64))
        .clamp(0.0, ((1u64 << 26) - 1) as f64) as u64;
    let lat_n = ((lat - GEO_LAT_MIN) / (GEO_LAT_MAX - GEO_LAT_MIN) * ((1u64 << 26) as f64))
        .clamp(0.0, ((1u64 << 26) - 1) as f64) as u64;
    let mut hash = 0u64;
    for b in (0..26).rev() {
        hash = (hash << 1) | ((lon_n >> b) & 1);
        hash = (hash << 1) | ((lat_n >> b) & 1);
    }
    Some(hash)
}

fn geohash_decode(hash: u64) -> (f64, f64) {
    let (mut lon_i, mut lat_i) = (0u64, 0u64);
    for b in (0..26).rev() {
        lon_i = (lon_i << 1) | ((hash >> (b * 2 + 1)) & 1);
        lat_i = (lat_i << 1) | ((hash >> (b * 2)) & 1);
    }
    let step = (1u64 << 26) as f64;
    let lon = GEO_LON_MIN + (lon_i as f64 + 0.5) / step * (GEO_LON_MAX - GEO_LON_MIN);
    let lat = GEO_LAT_MIN + (lat_i as f64 + 0.5) / step * (GEO_LAT_MAX - GEO_LAT_MIN);
    (lon, lat)
}

fn haversine_m(lon1: f64, lat1: f64, lon2: f64, lat2: f64) -> f64 {
    let (r1, r2) = (lat1.to_radians(), lat2.to_radians());
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2) + r1.cos() * r2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_M * a.sqrt().asin()
}

fn unit_scale_m(u: &str) -> Result<f64, RespVal> {
    match u.to_ascii_lowercase().as_str() {
        "m" => Ok(1.0),
        "km" => Ok(1000.0),
        "mi" => Ok(1609.344),
        "ft" => Ok(0.3048),
        _ => Err(RespVal::err(
            "ERR unsupported unit provided. please use M, KM, FT, MI",
        )),
    }
}

fn dispatch_geoadd(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 5 || !(args.len() - 2).is_multiple_of(3) {
        return RespVal::err("ERR wrong number of arguments for 'geoadd'");
    }
    let rec = match eng.ensure_entry_mut(&args[1], || Value::ZSet(BTreeMap::new())) {
        Ok(r) => r,
        Err(e) => return RespVal::err(e),
    };
    let Some(Value::ZSet(z)) = rec.value.as_mut() else {
        return RespVal::err(WRONGTYPE);
    };
    let mut added = 0i64;
    for tri in args[2..].as_chunks::<3>().0 {
        let (Ok(lon), Ok(lat)) = (parse_f64(&tri[0]), parse_f64(&tri[1])) else {
            return RespVal::err("ERR invalid longitude/latitude");
        };
        let Some(gh) = geohash_encode(lon, lat) else {
            return RespVal::err("ERR invalid longitude,latitude pair");
        };
        if z.insert(tri[2].clone(), gh as f64).is_none() {
            added += 1;
        }
    }
    RespVal::int(added)
}

fn dispatch_geopos(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 2 {
        return RespVal::err("ERR wrong number of arguments for 'geopos'");
    }
    let z_opt = match eng.get_record_or_load(&args[1]) {
        Ok(None) => None,
        Ok(Some(r)) => match r.value.as_ref() {
            Some(Value::ZSet(z)) => Some(z),
            _ => return RespVal::err(WRONGTYPE),
        },
        Err(e) => return RespVal::err(e),
    };
    let out = args[2..]
        .iter()
        .map(|m| match z_opt.and_then(|z| z.get(m)) {
            Some(&sc) => {
                let (lon, lat) = geohash_decode(sc as u64);
                RespVal::array(vec![
                    RespVal::bulk(format!("{lon:.6}").into_bytes()),
                    RespVal::bulk(format!("{lat:.6}").into_bytes()),
                ])
            }
            None => RespVal::nil_array(),
        })
        .collect();
    RespVal::array(out)
}

fn dispatch_geodist(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 4 || args.len() > 5 {
        return RespVal::err("ERR wrong number of arguments for 'geodist'");
    }
    let scale = match unit_scale_m(arg_str(args, 4).unwrap_or("m")) {
        Ok(s) => s,
        Err(e) => return e,
    };
    let Some(rec) = (match eng.get_record_or_load(&args[1]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    }) else {
        return RespVal::nil();
    };
    let Some(Value::ZSet(z)) = rec.value.as_ref() else {
        return RespVal::err(WRONGTYPE);
    };
    let (Some(&s1), Some(&s2)) = (z.get(&args[2]), z.get(&args[3])) else {
        return RespVal::nil();
    };
    let (lo1, la1) = geohash_decode(s1 as u64);
    let (lo2, la2) = geohash_decode(s2 as u64);
    let d = haversine_m(lo1, la1, lo2, la2) / scale;
    RespVal::bulk(format!("{d:.4}").into_bytes())
}

fn dispatch_geosearch(eng: &mut KvEngine, args: &[Vec<u8>]) -> RespVal {
    if args.len() < 6 {
        return RespVal::err("ERR wrong number of arguments for 'geosearch'");
    }
    let Some(rec) = (match eng.get_record_or_load(&args[1]) {
        Ok(v) => v,
        Err(e) => return RespVal::err(e),
    }) else {
        return RespVal::array(Vec::new());
    };
    let Some(Value::ZSet(z)) = rec.value.as_ref() else {
        return RespVal::err(WRONGTYPE);
    };
    let (mut center, mut radius_m, mut desc, mut count, mut withdist, mut unit_m) =
        (None, None, false, usize::MAX, false, 1.0);
    let mut i = 2;
    while i < args.len() {
        let tok = arg_str(args, i).unwrap_or("").to_ascii_uppercase();
        match tok.as_str() {
            "FROMMEMBER" if i + 1 < args.len() => {
                let Some(&sc) = z.get(&args[i + 1]) else {
                    return RespVal::err("ERR could not decode requested zset member");
                };
                center = Some(geohash_decode(sc as u64));
                i += 2;
            }
            "FROMLONLAT" if i + 2 < args.len() => {
                let (Ok(lo), Ok(la)) = (parse_f64(&args[i + 1]), parse_f64(&args[i + 2])) else {
                    return RespVal::err("ERR invalid lon/lat");
                };
                center = Some((lo, la));
                i += 3;
            }
            "BYRADIUS" if i + 2 < args.len() => {
                let Ok(r) = parse_f64(&args[i + 1]) else {
                    return RespVal::err("ERR invalid radius");
                };
                let sc = match unit_scale_m(arg_str(args, i + 2).unwrap_or("m")) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
                unit_m = sc;
                radius_m = Some(r * sc);
                i += 3;
            }
            "ASC" => {
                desc = false;
                i += 1;
            }
            "DESC" => {
                desc = true;
                i += 1;
            }
            "WITHDIST" => {
                withdist = true;
                i += 1;
            }
            "COUNT" if i + 1 < args.len() => {
                count = parse_i64(&args[i + 1]).unwrap_or(0).max(0) as usize;
                i += 2;
            }
            _ => i += 1,
        }
    }
    let (Some((clo, cla)), Some(rad)) = (center, radius_m) else {
        return RespVal::err("ERR GEOSEARCH requires FROMMEMBER/FROMLONLAT and BYRADIUS");
    };
    let mut hits: Vec<(&[u8], f64)> = z
        .iter()
        .filter_map(|(m, &sc)| {
            let (lo, la) = geohash_decode(sc as u64);
            let d = haversine_m(clo, cla, lo, la);
            (d <= rad).then_some((m.as_slice(), d))
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1));
    if desc {
        hits.reverse();
    }
    let out = hits
        .into_iter()
        .take(count)
        .map(|(m, d)| {
            if withdist {
                RespVal::array(vec![
                    RespVal::bulk(m.to_vec()),
                    RespVal::bulk(format!("{:.4}", d / unit_m).into_bytes()),
                ])
            } else {
                RespVal::bulk(m.to_vec())
            }
        })
        .collect();
    RespVal::array(out)
}
