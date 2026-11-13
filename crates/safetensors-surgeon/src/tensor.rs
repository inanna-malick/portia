//! Validated 2D edits with bounded conversion buffers and verbatim prefix copies.

use crate::mmap::Reader;
use crate::{ColFill, RowFill, SurgeonError, TensorEdit};
use half::{bf16, f16};
use safetensors::{tensor::TensorInfo, Dtype};
use std::io::Write;
use std::path::Path;

pub(crate) const BUFFER_BYTES: usize = 64 * 1024;
static ZEROES: [u8; BUFFER_BYTES] = [0; BUFFER_BYTES];

fn invalid(path: &Path, name: &str, message: &str) -> SurgeonError {
    SurgeonError::SafetensorsFormat {
        path: path.to_owned(),
        message: format!("tensor {name:?}: {message}"),
    }
}

fn nonfinite(name: &str, row: usize, col: usize) -> SurgeonError {
    SurgeonError::NonFiniteResult {
        name: name.to_owned(),
        row,
        col,
    }
}

fn encode(dtype: Dtype, value: f32) -> ([u8; 4], f32) {
    let bits = match dtype {
        Dtype::BF16 => u32::from(bf16::from_f32(value).to_bits()),
        Dtype::F16 => u32::from(f16::from_f32(value).to_bits()),
        Dtype::F32 => value.to_bits(),
        _ => unreachable!("only validated float dtypes are encoded"),
    };
    let bytes = bits.to_le_bytes();
    (bytes, decode(dtype, &bytes))
}

fn encoded_finite(
    name: &str,
    dtype: Dtype,
    value: f32,
    row: usize,
    col: usize,
) -> Result<[u8; 4], SurgeonError> {
    let (bytes, converted) = encode(dtype, value);
    if !value.is_finite() || !converted.is_finite() {
        return Err(nonfinite(name, row, col));
    }
    Ok(bytes)
}

fn buffer_length(name: &str, expected: usize, actual: usize) -> Result<(), SurgeonError> {
    if expected != actual {
        return Err(SurgeonError::BufferLengthMismatch {
            name: name.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn validate_append(
    name: &str,
    info: &TensorInfo,
    path: &Path,
    axis: usize,
    added: usize,
    values: Option<&[f32]>,
) -> Result<(usize, usize), SurgeonError> {
    if added == 0 {
        buffer_length(name, 1, 0)?;
    }
    let mut shape = [info.shape[0], info.shape[1]];
    let overflow = || invalid(path, name, "dimension or byte-length arithmetic overflow");
    shape[axis] = shape[axis].checked_add(added).ok_or_else(overflow)?;
    if let Some(values) = values {
        let expected = added.checked_mul(shape[1 - axis]).ok_or_else(overflow)?;
        buffer_length(name, expected, values.len())?;
        let (stride, row_offset, col_offset) = if axis == 0 {
            (shape[1], info.shape[0], 0)
        } else {
            (added, 0, info.shape[1])
        };
        for (index, &value) in values.iter().enumerate() {
            let row = row_offset + index / stride;
            let col = col_offset + index % stride;
            encoded_finite(name, info.dtype, value, row, col)?;
        }
    }
    Ok((shape[0], shape[1]))
}

/// Return the output shape and its byte length as offsets `(0, length)`.
pub(crate) fn validate(
    name: &str,
    info: &TensorInfo,
    edit: &TensorEdit,
    path: &Path,
) -> Result<TensorInfo, SurgeonError> {
    let [rows, cols] = info.shape.as_slice() else {
        return Err(SurgeonError::NotMatrix2D {
            name: name.to_owned(),
            actual: info.shape.clone(),
        });
    };
    let (rows, cols) = (*rows, *cols);
    if !matches!(info.dtype, Dtype::BF16 | Dtype::F16 | Dtype::F32) {
        return Err(SurgeonError::UnsupportedDtype {
            name: name.into(),
            dtype: info.dtype,
        });
    }
    let overflow = || invalid(path, name, "dimension or byte-length arithmetic overflow");
    let (new_rows, new_cols) = match edit {
        TensorEdit::AddDense2D {
            rows: expected_rows,
            cols: expected_cols,
            delta_row_major,
        } => {
            if (rows, cols) != (*expected_rows, *expected_cols) {
                return Err(SurgeonError::DimensionMismatch {
                    name: name.to_owned(),
                    expected: (*expected_rows, *expected_cols),
                    actual: (rows, cols),
                });
            }
            buffer_length(
                name,
                rows.checked_mul(cols).ok_or_else(overflow)?,
                delta_row_major.len(),
            )?;
            for (index, value) in delta_row_major.iter().enumerate() {
                if !value.is_finite() {
                    return Err(nonfinite(name, index / cols, index % cols));
                }
            }
            (rows, cols)
        }
        TensorEdit::AppendRows2D { fill } => {
            let (added, values) = match fill {
                RowFill::Zeros { added_rows } => (*added_rows, None),
                RowFill::Values { added_rows, data } => (*added_rows, Some(data.as_slice())),
            };
            validate_append(name, info, path, 0, added, values)?
        }
        TensorEdit::AppendCols2D { fill } => {
            let (added, values) = match fill {
                ColFill::Zeros { added_cols } => (*added_cols, None),
                ColFill::Values { added_cols, data } => (*added_cols, Some(data.as_slice())),
            };
            validate_append(name, info, path, 1, added, values)?
        }
    };
    let length = new_rows
        .checked_mul(new_cols)
        .and_then(|n| n.checked_mul(info.dtype.bitsize()))
        .map(|bits| bits / 8)
        .ok_or_else(overflow)?;
    Ok(TensorInfo {
        dtype: info.dtype,
        shape: vec![new_rows, new_cols],
        data_offsets: (0, length),
    })
}

fn output(writer: &mut dyn Write, bytes: &[u8], path: &Path) -> Result<(), SurgeonError> {
    writer.write_all(bytes).map_err(|source| SurgeonError::Io {
        path: path.to_owned(),
        source,
    })
}

fn zeros(mut length: usize, writer: &mut dyn Write, path: &Path) -> Result<(), SurgeonError> {
    while length != 0 {
        let count = length.min(ZEROES.len());
        output(writer, &ZEROES[..count], path)?;
        length -= count;
    }
    Ok(())
}

fn values(
    dtype: Dtype,
    data: &[f32],
    buffer: &mut [u8],
    writer: &mut dyn Write,
    path: &Path,
) -> Result<(), SurgeonError> {
    let size = if dtype == Dtype::F32 { 4 } else { 2 };
    for chunk in data.chunks(buffer.len() / size) {
        for (bytes, &value) in buffer.chunks_exact_mut(size).zip(chunk) {
            bytes.copy_from_slice(&encode(dtype, value).0[..size]);
        }
        output(writer, &buffer[..chunk.len() * size], path)?;
    }
    Ok(())
}

fn decode(dtype: Dtype, bytes: &[u8]) -> f32 {
    match dtype {
        Dtype::BF16 => bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        Dtype::F16 => f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32(),
        Dtype::F32 => f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => unreachable!("only validated float dtypes are decoded"),
    }
}

/// Stream an edit after `validate`; no buffer grows with a tensor or row size.
pub(crate) fn write(
    name: &str,
    info: &TensorInfo,
    reader: &mut Reader<'_>,
    data_start: u64,
    edit: &TensorEdit,
    buffer: &mut [u8],
    writer: &mut dyn Write,
) -> Result<(), SurgeonError> {
    let path = reader.path();
    let size = info.dtype.bitsize() / 8;
    let rows = info.shape[0];
    let cols = info.shape[1];
    let start = data_start
        .checked_add(info.data_offsets.0 as u64)
        .ok_or_else(|| invalid(path, name, "input data offset overflow"))?;
    match edit {
        TensorEdit::AddDense2D {
            delta_row_major, ..
        } => {
            if rows == 0 || cols == 0 {
                return Ok(());
            }
            for row in 0..rows {
                let mut col = 0;
                while col < cols {
                    let count = (cols - col).min(buffer.len() / size);
                    let index = row * cols + col;
                    reader
                        .read_exact(start + (index * size) as u64, &mut buffer[..count * size])?;
                    for (index_in_chunk, bytes) in
                        buffer[..count * size].chunks_exact_mut(size).enumerate()
                    {
                        let value =
                            decode(info.dtype, bytes) + delta_row_major[index + index_in_chunk];
                        let encoded =
                            encoded_finite(name, info.dtype, value, row, col + index_in_chunk)?;
                        bytes.copy_from_slice(&encoded[..size]);
                    }
                    output(writer, &buffer[..count * size], path)?;
                    col += count;
                }
            }
        }
        TensorEdit::AppendRows2D { fill } => {
            reader.copy_range(start, info.data_offsets.1 - info.data_offsets.0, writer)?;
            match fill {
                RowFill::Zeros { added_rows } => zeros(added_rows * cols * size, writer, path)?,
                RowFill::Values { data, .. } => values(info.dtype, data, buffer, writer, path)?,
            }
        }
        TensorEdit::AppendCols2D { fill } => {
            for row in 0..rows {
                reader.copy_range(start + (row * cols * size) as u64, cols * size, writer)?;
                match fill {
                    ColFill::Zeros { added_cols } => zeros(added_cols * size, writer, path)?,
                    ColFill::Values { added_cols, data } => {
                        let appended = &data[row * added_cols..(row + 1) * added_cols];
                        values(info.dtype, appended, buffer, writer, path)?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(dtype: Dtype, values: &[f32]) -> Vec<u8> {
        let size = dtype.bitsize() / 8;
        values
            .iter()
            .flat_map(|&v| encode(dtype, v).0[..size].to_vec())
            .collect()
    }

    fn info(dtype: Dtype, rows: usize, cols: usize) -> TensorInfo {
        TensorInfo {
            dtype,
            shape: vec![rows, cols],
            data_offsets: (0, rows * cols * dtype.bitsize() / 8),
        }
    }

    fn run(
        info: &TensorInfo,
        input: &[u8],
        edit: &TensorEdit,
    ) -> Result<(TensorInfo, Vec<u8>), SurgeonError> {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(input).unwrap();
        let updated = validate("w", info, edit, file.path())?;
        let mut output = Vec::new();
        let mut reader = Reader::new(file.as_file(), file.path())?;
        let mut buffer = vec![0; BUFFER_BYTES];
        write("w", info, &mut reader, 0, edit, &mut buffer, &mut output)?;
        Ok((updated, output))
    }

    #[test]
    fn all_float_edits_preserve_raw_prefixes() {
        for dtype in [Dtype::BF16, Dtype::F16, Dtype::F32] {
            let info = info(dtype, 2, 2);
            let input = encoded(dtype, &[1.0, -2.0, 3.5, 4.0]);
            let edit = TensorEdit::AddDense2D {
                rows: 2,
                cols: 2,
                delta_row_major: vec![1.0; 4],
            };
            assert_eq!(
                run(&info, &input, &edit).unwrap().1,
                encoded(dtype, &[2.0, -1.0, 4.5, 5.0])
            );
            let input = encoded(dtype, &[f32::NAN, -0.0, f32::INFINITY, -0.0]);
            let edit = TensorEdit::AppendRows2D {
                fill: RowFill::Values {
                    added_rows: 1,
                    data: vec![7.0, 8.0],
                },
            };
            let (updated, output) = run(&info, &input, &edit).unwrap();
            assert_eq!(updated.shape, [3, 2]);
            assert_eq!(&output[..input.len()], input);
            assert_eq!(&output[input.len()..], encoded(dtype, &[7.0, 8.0]));
            let edit = TensorEdit::AppendCols2D {
                fill: ColFill::Values {
                    added_cols: 1,
                    data: vec![7.0, 8.0],
                },
            };
            let (updated, output) = run(&info, &input, &edit).unwrap();
            assert_eq!(updated.shape, [2, 3]);
            assert_eq!(
                output,
                encoded(dtype, &[f32::NAN, -0.0, 7.0, f32::INFINITY, -0.0, 8.0])
            );
        }
    }

    #[test]
    fn zero_dimensions_and_zero_growth_are_checked() {
        let zero_rows = info(Dtype::F32, 0, 2);
        let edit = TensorEdit::AppendCols2D {
            fill: ColFill::Zeros { added_cols: 3 },
        };
        let (updated, output) = run(&zero_rows, &[], &edit).unwrap();
        assert_eq!(updated.shape, [0, 5]);
        assert!(output.is_empty());
        let zero_cols = info(Dtype::F32, 2, 0);
        let (updated, output) = run(&zero_cols, &[], &edit).unwrap();
        assert_eq!(updated.shape, [2, 3]);
        assert_eq!(output, vec![0; 24]);
        for edit in [
            TensorEdit::AppendRows2D {
                fill: RowFill::Zeros { added_rows: 0 },
            },
            TensorEdit::AppendCols2D {
                fill: ColFill::Zeros { added_cols: 0 },
            },
        ] {
            assert!(matches!(
                validate("w", &zero_rows, &edit, Path::new("w")),
                Err(SurgeonError::BufferLengthMismatch { .. })
            ));
        }
    }

    #[test]
    fn input_and_result_overflow_report_coordinates() {
        for dtype in [Dtype::BF16, Dtype::F16, Dtype::F32] {
            let info = info(dtype, 1, 1);
            let max = match dtype {
                Dtype::BF16 => bf16::MAX.to_f32(),
                Dtype::F16 => f16::MAX.to_f32(),
                _ => f32::MAX,
            };
            let edit = TensorEdit::AddDense2D {
                rows: 1,
                cols: 1,
                delta_row_major: vec![max],
            };
            assert!(matches!(
                run(&info, &encoded(dtype, &[max]), &edit),
                Err(SurgeonError::NonFiniteResult { row: 0, col: 0, .. })
            ));
            let edit = TensorEdit::AppendCols2D {
                fill: ColFill::Values {
                    added_cols: 1,
                    data: vec![f32::NAN],
                },
            };
            assert!(matches!(
                validate("w", &info, &edit, Path::new("w")),
                Err(SurgeonError::NonFiniteResult { row: 0, col: 1, .. })
            ));
        }
    }
}
