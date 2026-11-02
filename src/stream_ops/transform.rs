use std::io::{Read, Write};

use crate::{
    ops::{validate_projection_cols, ColOp, ColPred, ColTarget, PredExpr},
    DecodeOptions, Error, Frame, Options, Result,
};

use super::{write_all_vxg, FramePeeler, FrameReader};

fn write_non_empty_frames<W: Write>(
    writer: &mut W,
    encoded: &[u8],
    frames_written: &mut u64,
    bytes_written: &mut u64,
) -> Result<()> {
    let parsed = Frame::parse_all(encoded, usize::MAX)?;
    if parsed.iter().all(|f| f.raw_size() == 0) {
        return Ok(());
    }
    write_all_vxg(writer, encoded)?;
    *frames_written += parsed.len() as u64;
    *bytes_written += encoded.len() as u64;
    Ok(())
}

fn finish_empty_frame_if_needed<W: Write>(
    writer: &mut W,
    enc: &Options,
    frames_written: u64,
    bytes_written: &mut u64,
) -> Result<()> {
    if frames_written == 0 {
        let empty = crate::compress(&[], enc)?;
        write_all_vxg(writer, &empty)?;
        *bytes_written += empty.len() as u64;
    }
    writer.flush().map_err(|e| Error::Backend(e.to_string()))?;
    Ok(())
}

/// Bounded-memory streaming column projection into `VXG1` frames (`R: Read` -> `W: Write`),
/// using zero-reencode compressed column splicing on interior `GRAMMAR_PACKET` blocks and
/// exact cross-frame boundary stitching. Returns total compressed bytes written.
pub fn project_frame_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    where_expr: Option<&PredExpr<'_>>,
    cols: &[ColTarget<'_>],
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    enc_options.validate()?;
    dec_options.validate()?;
    validate_projection_cols(cols)?;

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut frames_written = 0u64;
    let mut bytes_written = 0u64;
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let proj = crate::project_frame(eff, where_expr, cols, enc_options, &sub_dec)?;
            write_non_empty_frames(&mut writer, &proj, &mut frames_written, &mut bytes_written)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let proj = crate::project_frame(tail_eff, where_expr, cols, enc_options, &sub_dec)?;
        write_non_empty_frames(&mut writer, &proj, &mut frames_written, &mut bytes_written)?;
    }
    finish_empty_frame_if_needed(&mut writer, enc_options, frames_written, &mut bytes_written)?;
    Ok(bytes_written)
}

/// Bounded-memory streaming compound `WHERE` frame filter (`R: Read` -> `W: Write`),
/// splicing 100%-matching compressed blocks verbatim. Returns total compressed bytes written.
pub fn filter_frame_where_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    expr: &PredExpr<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    enc_options.validate()?;
    dec_options.validate()?;

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut frames_written = 0u64;
    let mut bytes_written = 0u64;
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let out = crate::filter_frame_where(eff, expr, enc_options, &sub_dec)?;
            write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let out = crate::filter_frame_where(tail_eff, expr, enc_options, &sub_dec)?;
        write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
    }
    finish_empty_frame_if_needed(&mut writer, enc_options, frames_written, &mut bytes_written)?;
    Ok(bytes_written)
}

/// Bounded-memory streaming conditional single-column update (`R: Read` -> `W: Write`),
/// preserving 0-hit blocks verbatim and splicing untouched columns. Returns compressed bytes written.
pub fn update_column_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    where_pred: &ColPred<'_>,
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    enc_options.validate()?;
    dec_options.validate()?;

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut frames_written = 0u64;
    let mut bytes_written = 0u64;
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let out = crate::update_column(eff, where_pred, target_col, op, enc_options, &sub_dec)?;
            write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let out =
            crate::update_column(tail_eff, where_pred, target_col, op, enc_options, &sub_dec)?;
        write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
    }
    finish_empty_frame_if_needed(&mut writer, enc_options, frames_written, &mut bytes_written)?;
    Ok(bytes_written)
}

/// Bounded-memory streaming unconditional column map (`R: Read` -> `W: Write`),
/// transforming dictionary columns in `O(|D|)` per block. Returns compressed bytes written.
pub fn map_column_stream<R: Read, W: Write>(
    reader: R,
    mut writer: W,
    target_col: ColTarget<'_>,
    op: ColOp<'_>,
    enc_options: &Options,
    dec_options: &DecodeOptions,
) -> Result<u64> {
    enc_options.validate()?;
    dec_options.validate()?;

    let mut fr = FrameReader::new(reader, dec_options.max_output);
    let mut peeler = FramePeeler::new();
    let mut frames_written = 0u64;
    let mut bytes_written = 0u64;
    let mut sub_dec = dec_options.clone();
    sub_dec.max_output = usize::MAX;

    while let Some(f_bytes) = fr.next_frame()? {
        if let Some(eff) = peeler.peel_frame(f_bytes, &sub_dec)? {
            let out = crate::map_column(eff, target_col, op, enc_options, &sub_dec)?;
            write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
        }
    }
    if let Some(tail_eff) = peeler.finish_tail()? {
        let out = crate::map_column(tail_eff, target_col, op, enc_options, &sub_dec)?;
        write_non_empty_frames(&mut writer, &out, &mut frames_written, &mut bytes_written)?;
    }
    finish_empty_frame_if_needed(&mut writer, enc_options, frames_written, &mut bytes_written)?;
    Ok(bytes_written)
}
