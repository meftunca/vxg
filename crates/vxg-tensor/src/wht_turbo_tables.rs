//! Pre-fitted positive-orthant & coordinate-sparse Gaussian vector quantization codebooks
//! (`Orth4D-16`, `Sym64-4D`, `Orth2D-16`, and `EC-Orth2D-16`) stored in exact IEEE-754 `f16` bits.

use crate::fp16::f16_to_f32;

/// Optimal 16-centroid 4D positive-orthant Gaussian codebook (`Turbo2`, `2.125 bpw`, `~10.03 dB`).
pub const ORTH4D_16_F16: [[u16; 4]; 16] = [
    [0x35c4, 0x35e9, 0x35ca, 0x35cb],
    [0x3534, 0x3cbe, 0x377b, 0x37b3],
    [0x3cc3, 0x3540, 0x37a5, 0x3757],
    [0x374a, 0x3738, 0x372b, 0x3cd3],
    [0x3757, 0x372f, 0x3cd8, 0x3726],
    [0x3c7c, 0x3c7e, 0x3741, 0x379f],
    [0x38af, 0x38a2, 0x3e16, 0x3e2d],
    [0x3885, 0x3e2c, 0x3e2d, 0x38be],
    [0x3e2c, 0x38af, 0x38bf, 0x3e31],
    [0x3e47, 0x38cd, 0x3e30, 0x38be],
    [0x38b5, 0x3e30, 0x38de, 0x3e62],
    [0x3868, 0x4071, 0x385c, 0x3865],
    [0x408d, 0x389f, 0x385c, 0x3885],
    [0x38a9, 0x3866, 0x3893, 0x4094],
    [0x38b6, 0x3885, 0x4097, 0x38d9],
    [0x3f25, 0x3f5e, 0x3a22, 0x3a06],
];

/// Permutation-symmetric 64-centroid 4D ECVQ codebook (`EC-Turbo2.5`, `bits = 5`) with origin,
/// axis (`nnz = 1`), triple (`nnz = 3`), and dense (`nnz = 4`) orbits + zero-coordinate sign elision.
pub const ORTH4D_64_F16: [[u16; 4]; 64] = [
    [0x0000, 0x0000, 0x0000, 0x0000],
    [0x3b24, 0x0000, 0x0000, 0x0000],
    [0x0000, 0x3b24, 0x0000, 0x0000],
    [0x0000, 0x0000, 0x3b24, 0x0000],
    [0x0000, 0x0000, 0x0000, 0x3b24],
    [0x0000, 0x3c6a, 0x3c6a, 0x3c6a],
    [0x3c6a, 0x0000, 0x3c6a, 0x3c6a],
    [0x3c6a, 0x3c6a, 0x0000, 0x3c6a],
    [0x3c6a, 0x3c6a, 0x3c6a, 0x0000],
    [0x3645, 0x3645, 0x3645, 0x3645],
    [0x3ad0, 0x3ad0, 0x3ad0, 0x3ad0],
    [0x3dba, 0x3dba, 0x3dba, 0x3dba],
    [0x3ded, 0x3603, 0x3603, 0x3603],
    [0x3603, 0x3ded, 0x3603, 0x3603],
    [0x3603, 0x3603, 0x3ded, 0x3603],
    [0x3603, 0x3603, 0x3603, 0x3ded],
    [0x40b9, 0x3602, 0x3602, 0x3602],
    [0x3602, 0x40b9, 0x3602, 0x3602],
    [0x3602, 0x3602, 0x40b9, 0x3602],
    [0x3602, 0x3602, 0x3602, 0x40b9],
    [0x41b7, 0x3a89, 0x3a89, 0x3a89],
    [0x3a89, 0x41b7, 0x3a89, 0x3a89],
    [0x3a89, 0x3a89, 0x41b7, 0x3a89],
    [0x3a89, 0x3a89, 0x3a89, 0x41b7],
    [0x355a, 0x3e7b, 0x3e7b, 0x3e7b],
    [0x3e7b, 0x355a, 0x3e7b, 0x3e7b],
    [0x3e7b, 0x3e7b, 0x355a, 0x3e7b],
    [0x3e7b, 0x3e7b, 0x3e7b, 0x355a],
    [0x3b89, 0x3b89, 0x351a, 0x351a],
    [0x3b89, 0x351a, 0x3b89, 0x351a],
    [0x3b89, 0x351a, 0x351a, 0x3b89],
    [0x351a, 0x3b89, 0x3b89, 0x351a],
    [0x351a, 0x3b89, 0x351a, 0x3b89],
    [0x351a, 0x351a, 0x3b89, 0x3b89],
    [0x3d9b, 0x3d9b, 0x3a4d, 0x3a4d],
    [0x3d9b, 0x3a4d, 0x3d9b, 0x3a4d],
    [0x3d9b, 0x3a4d, 0x3a4d, 0x3d9b],
    [0x3a4d, 0x3d9b, 0x3d9b, 0x3a4d],
    [0x3a4d, 0x3d9b, 0x3a4d, 0x3d9b],
    [0x3a4d, 0x3a4d, 0x3d9b, 0x3d9b],
    [0x403f, 0x403f, 0x3780, 0x3780],
    [0x403f, 0x3780, 0x403f, 0x3780],
    [0x403f, 0x3780, 0x3780, 0x403f],
    [0x3780, 0x403f, 0x403f, 0x3780],
    [0x3780, 0x403f, 0x3780, 0x403f],
    [0x3780, 0x3780, 0x403f, 0x403f],
    [0x4029, 0x4029, 0x3c68, 0x3c68],
    [0x4029, 0x3c68, 0x4029, 0x3c68],
    [0x4029, 0x3c68, 0x3c68, 0x4029],
    [0x3c68, 0x4029, 0x4029, 0x3c68],
    [0x3c68, 0x4029, 0x3c68, 0x4029],
    [0x3c68, 0x3c68, 0x4029, 0x4029],
    [0x3f7c, 0x3c98, 0x3698, 0x3698],
    [0x3f7c, 0x3698, 0x3c98, 0x3698],
    [0x3f7c, 0x3698, 0x3698, 0x3c98],
    [0x3c98, 0x3f7c, 0x3698, 0x3698],
    [0x3698, 0x3f7c, 0x3c98, 0x3698],
    [0x3698, 0x3f7c, 0x3698, 0x3c98],
    [0x3c98, 0x3698, 0x3f7c, 0x3698],
    [0x3698, 0x3c98, 0x3f7c, 0x3698],
    [0x3698, 0x3698, 0x3f7c, 0x3c98],
    [0x3c98, 0x3698, 0x3698, 0x3f7c],
    [0x3698, 0x3c98, 0x3698, 0x3f7c],
    [0x3698, 0x3698, 0x3c98, 0x3f7c],
];

/// Empirical Shannon self-information + true zero-elided sign bit cost per `ORTH4D_64_F16` centroid.
pub(crate) const EC_BITS_4D_64: [f32; 64] = [
    4.200, 7.373, 7.373, 7.373, 7.373, 9.022, 9.022, 9.022,
    9.022, 7.737, 9.594, 12.057, 8.866, 8.866, 8.866, 8.866,
    10.735, 10.735, 10.735, 10.735, 11.338, 11.338, 11.338, 11.338,
    11.091, 11.091, 11.091, 11.091, 8.650, 8.650, 8.650, 8.650,
    8.650, 8.650, 10.474, 10.474, 10.474, 10.474, 10.474, 10.474,
    11.790, 11.790, 11.790, 11.790, 11.790, 11.790, 12.615, 12.615,
    12.615, 12.615, 12.615, 12.615, 9.850, 9.850, 9.850, 9.850,
    9.850, 9.850, 9.850, 9.850, 9.850, 9.850, 9.850, 9.850,
];

/// Optimal 16-centroid 2D positive-quadrant Gaussian codebook (`Turbo3`, `3.125 bpw`, `~15.36 dB`).
pub const ORTH2D_16_F16: [[u16; 2]; 16] = [
    [0x3297, 0x32aa],
    [0x390d, 0x33cf],
    [0x331b, 0x3932],
    [0x3973, 0x39fd],
    [0x3c6c, 0x33ee],
    [0x33e9, 0x3ca3],
    [0x3cbc, 0x3a20],
    [0x3a26, 0x3d4a],
    [0x3ec5, 0x3473],
    [0x34f5, 0x3f17],
    [0x3d68, 0x3d72],
    [0x3f4d, 0x3b2e],
    [0x3c80, 0x4022],
    [0x4113, 0x37ae],
    [0x37ab, 0x4138],
    [0x4052, 0x3f10],
];

/// Permutation-symmetric 16-centroid 2D ECVQ codebook (`EC-Turbo3`, `bits = 4`) with exact `(0, 0)` origin.
pub const EC_ORTH2D_16_F16: [[u16; 2]; 16] = [
    [0x0000, 0x0000],
    [0x3981, 0x346c],
    [0x346c, 0x3981],
    [0x3b48, 0x3b48],
    [0x3d57, 0x352a],
    [0x352a, 0x3d57],
    [0x3e61, 0x3c01],
    [0x3c01, 0x3e61],
    [0x4014, 0x35c7],
    [0x35c7, 0x4014],
    [0x3edb, 0x3edb],
    [0x40cc, 0x3cc9],
    [0x3cc9, 0x40cc],
    [0x41c2, 0x3726],
    [0x3726, 0x41c2],
    [0x409e, 0x409e],
];

/// Empirical Shannon self-information + true zero-elided sign bit cost per `EC_ORTH2D_16_F16` centroid.
pub(crate) const EC_BITS_2D_16: [f32; 16] = [
    3.673, 4.527, 4.527, 5.103, 5.222, 5.222, 6.250, 6.250,
    6.668, 6.668, 7.856, 8.310, 8.310, 9.135, 9.135, 10.197,
];

#[inline(always)]
pub(crate) fn ec_orth2d_nz_mask(idx: usize) -> u8 {
    let c = EC_ORTH2D_16_F16[idx & 15];
    ((c[0] != 0) as u8) | (((c[1] != 0) as u8) << 1)
}

#[inline(always)]
pub(crate) fn orth4d64_nz_mask(idx: usize) -> u8 {
    let c = ORTH4D_64_F16[idx & 63];
    ((c[0] != 0) as u8)
        | (((c[1] != 0) as u8) << 1)
        | (((c[2] != 0) as u8) << 2)
        | (((c[3] != 0) as u8) << 3)
}

#[inline(always)]
pub(crate) fn unpack_t25_lane(p80: &[u32], lane: usize) -> [[usize; 2]; 4] {
    let w0 = p80[lane * 2];
    let w1 = p80[lane * 2 + 1];
    let h16 = (p80[64 + (lane >> 1)] >> ((lane & 1) * 16)) & 0xffff;
    [
        [(w0 & 1023) as usize, ((w0 >> 10) & 1023) as usize],
        [((w0 >> 20) & 1023) as usize, (w1 & 1023) as usize],
        [((w1 >> 10) & 1023) as usize, ((w1 >> 20) & 1023) as usize],
        [
            ((h16 & 0xff) | ((w0 >> 30) << 8)) as usize,
            ((h16 >> 8) | ((w1 >> 30) << 8)) as usize,
        ],
    ]
}

pub(crate) fn orth4d_tables() -> ([[f32; 4]; 16], [f32; 16], [[f32; 4]; 256]) {
    let mut cb = [[0.0f32; 4]; 16];
    let mut n2 = [0.0f32; 16];
    for i in 0..16 {
        for k in 0..4 {
            cb[i][k] = f16_to_f32(ORTH4D_16_F16[i][k]);
        }
        n2[i] = 0.5
            * (cb[i][0] * cb[i][0]
                + cb[i][1] * cb[i][1]
                + cb[i][2] * cb[i][2]
                + cb[i][3] * cb[i][3]);
    }
    let mut lut256 = [[0.0f32; 4]; 256];
    for b in 0..256usize {
        let base = ORTH4D_16_F16[b & 0x0f];
        lut256[b] = [
            f16_to_f32(base[0] | (((b as u16) & 0x10) << 11)),
            f16_to_f32(base[1] | (((b as u16) & 0x20) << 10)),
            f16_to_f32(base[2] | (((b as u16) & 0x40) << 9)),
            f16_to_f32(base[3] | (((b as u16) & 0x80) << 8)),
        ];
    }
    (cb, n2, lut256)
}

pub(crate) fn orth4d64_tables() -> ([[f32; 4]; 64], [f32; 64], Vec<[f32; 4]>) {
    let mut cb = [[0.0f32; 4]; 64];
    let mut n2 = [0.0f32; 64];
    for i in 0..64 {
        for k in 0..4 {
            cb[i][k] = f16_to_f32(ORTH4D_64_F16[i][k]);
        }
        n2[i] = 0.5
            * (cb[i][0] * cb[i][0]
                + cb[i][1] * cb[i][1]
                + cb[i][2] * cb[i][2]
                + cb[i][3] * cb[i][3]);
    }
    let mut lut1024 = vec![[0.0f32; 4]; 1024];
    for c10 in 0..1024usize {
        let base = ORTH4D_64_F16[c10 & 63];
        let s = (c10 >> 6) as u16;
        lut1024[c10] = [
            if base[0] == 0 { 0.0 } else { f16_to_f32(base[0] | ((s & 1) << 15)) },
            if base[1] == 0 { 0.0 } else { f16_to_f32(base[1] | ((s & 2) << 14)) },
            if base[2] == 0 { 0.0 } else { f16_to_f32(base[2] | ((s & 4) << 13)) },
            if base[3] == 0 { 0.0 } else { f16_to_f32(base[3] | ((s & 8) << 12)) },
        ];
    }
    (cb, n2, lut1024)
}

fn build_2d_tables(raw: &[[u16; 2]; 16]) -> ([[f32; 2]; 16], [f32; 16], [[f32; 2]; 64]) {
    let mut cb = [[0.0f32; 2]; 16];
    let mut n2 = [0.0f32; 16];
    for i in 0..16 {
        cb[i] = [f16_to_f32(raw[i][0]), f16_to_f32(raw[i][1])];
        n2[i] = 0.5 * (cb[i][0] * cb[i][0] + cb[i][1] * cb[i][1]);
    }
    let mut lut64 = [[0.0f32; 2]; 64];
    for b in 0..64usize {
        let base = raw[b & 0x0f];
        lut64[b] = [
            if base[0] == 0 { 0.0 } else { f16_to_f32(base[0] | (((b as u16) & 0x10) << 11)) },
            if base[1] == 0 { 0.0 } else { f16_to_f32(base[1] | (((b as u16) & 0x20) << 10)) },
        ];
    }
    (cb, n2, lut64)
}

pub(crate) fn orth2d_tables() -> ([[f32; 2]; 16], [f32; 16], [[f32; 2]; 64]) {
    build_2d_tables(&ORTH2D_16_F16)
}

pub(crate) fn ec_orth2d_tables() -> ([[f32; 2]; 16], [f32; 16], [[f32; 2]; 64]) {
    build_2d_tables(&EC_ORTH2D_16_F16)
}
