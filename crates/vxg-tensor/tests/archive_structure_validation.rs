use vxg_tensor::codebook::CodebookMatrix;
use vxg_tensor::format::{read_vxt_archive, write_vxt_archive, PackedTensor, VxtEntry};
use vxg_tensor::lossless::LosslessMatrix;
use vxg_tensor::wht2::Wht2Matrix;
use vxg_tensor::{DType, Error};

// Serialize the malformed structure through the writer so BOTH checksums are valid.
fn rejects(tensor: PackedTensor) {
    let bytes = write_vxt_archive(&[VxtEntry {
        name: "w".into(),
        tensor,
    }])
    .unwrap();
    assert!(matches!(read_vxt_archive(&bytes), Err(Error::Format(_))));
}

fn lossless(dtype: DType) -> LosslessMatrix {
    let words: Vec<u16> = (0..513)
        .map(|i| match dtype {
            DType::BF16 => ((i % 256) << 7) as u16,
            DType::F16 => ((i % 32) << 10) as u16,
        })
        .collect();
    LosslessMatrix::encode(dtype, 1, words.len(), &words).unwrap()
}

#[test]
fn lossless_rejects_corrupt_escape_streams_with_valid_checksums() {
    for dtype in [DType::BF16, DType::F16] {
        let good = lossless(dtype);
        assert!(!good.esc_bytes.is_empty());
        for offset in [1, u32::MAX] {
            let mut bad = good.clone();
            bad.sb_esc_offsets[0] = offset;
            rejects(PackedTensor::Lossless(bad));
        }
        let mut bad = good.clone();
        bad.sb_esc_offsets[1] = 0;
        rejects(PackedTensor::Lossless(bad));
        let mut bad = good.clone();
        bad.esc_bytes.pop();
        rejects(PackedTensor::Lossless(bad));
        let mut bad = good.clone();
        bad.esc_bytes.push(0);
        rejects(PackedTensor::Lossless(bad));
        // An escape in padding is consumed by GEMV/GPU even when decode_row skips it.
        let mut bad = good.clone();
        let start = if dtype == DType::BF16 { 0 } else { 24 };
        let wps = if dtype == DType::BF16 { 24 } else { 48 };
        for plane in 0..3 {
            bad.code_words[2 * wps + start + plane * 8 + 7] |= 1 << 31;
        }
        rejects(PackedTensor::Lossless(bad));
        let bytes = write_vxt_archive(&[VxtEntry {
            name: "w".into(),
            tensor: PackedTensor::Lossless(good.clone()),
        }])
        .unwrap();
        let parsed = read_vxt_archive(&bytes).unwrap();
        if let PackedTensor::Lossless(m) = &parsed[0].tensor {
            assert_eq!(m.decode_u16(), good.decode_u16());
        }
    }
}

#[test]
fn fp16_rejects_out_of_range_exponents() {
    let mut bad = lossless(DType::F16);
    bad.lut[0] = 32;
    rejects(PackedTensor::Lossless(bad));
    let mut bad = lossless(DType::F16);
    bad.esc_bytes[0] = 255;
    rejects(PackedTensor::Lossless(bad));
}

fn codebook(ptr: Vec<u32>, outliers: Vec<u32>) -> PackedTensor {
    PackedTensor::Codebook(CodebookMatrix {
        bits: 4,
        rows: 2,
        cols: 3,
        padded_cols: 256,
        centroids: [0.; 16],
        scales_f16: vec![0; 4],
        packed_words: vec![0; 64],
        outlier_row_ptr: ptr,
        outliers_packed: outliers,
    })
}
fn wht2(ptr: Vec<u32>, outliers: Vec<u32>) -> PackedTensor {
    PackedTensor::Wht2(Wht2Matrix {
        rows: 2,
        cols: 3,
        padded_cols: 256,
        centroids: [0.; 16],
        scales_f16: vec![0; 4],
        packed_words: vec![0; 32],
        outlier_row_ptr: ptr,
        outliers_packed: outliers,
    })
}
#[test]
fn outlier_csr_rejects_nonzero_start_nonmonotonic_and_invalid_columns() {
    for make in [codebook as fn(Vec<u32>, Vec<u32>) -> PackedTensor, wht2] {
        rejects(make(vec![1, 1, 1], vec![0]));
        rejects(make(vec![0, 2, 1], vec![0]));
        rejects(make(vec![0, u32::MAX, 1], vec![0]));
        rejects(make(vec![0, 0, 1], vec![3]));
        let bytes = write_vxt_archive(&[VxtEntry {
            name: "w".into(),
            tensor: make(vec![0, 0, 1], vec![2]),
        }])
        .unwrap();
        assert!(read_vxt_archive(&bytes).is_ok());
    }
}

#[test]
fn archive_dimensions_cannot_overflow_plane_size_arithmetic() {
    let mut bad = lossless(DType::BF16);
    bad.cols = usize::MAX;
    bad.padded_cols = usize::MAX;
    rejects(PackedTensor::Lossless(bad));
    let mut bad = lossless(DType::BF16);
    bad.rows = usize::MAX;
    rejects(PackedTensor::Lossless(bad));
}
