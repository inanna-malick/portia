use proptest::prelude::*;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use safetensors_surgeon::*;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs};

fn encode(dtype: Dtype, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| match dtype {
            Dtype::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
            Dtype::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
            _ => v.to_le_bytes().to_vec(),
        })
        .collect()
}
fn fixture(dir: &std::path::Path, dtype: Dtype, shape: Vec<usize>, bytes: &[u8]) {
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    let metadata = Some(HashMap::from([("origin".into(), "test".into())]));
    fs::write(
        dir.join("model.safetensors"),
        safetensors::serialize([("w", view)], metadata).unwrap(),
    )
    .unwrap();
}
fn run(dtype: Dtype, r: usize, c: usize, k: usize, mode: usize, raw: Vec<u8>) {
    let input = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("out");
    fixture(input.path(), dtype, vec![r, c], &raw);
    let edit = match mode {
        0 => TensorEdit::AddDense2D {
            rows: r,
            cols: c,
            delta_row_major: vec![0.25; r * c],
        },
        1 => TensorEdit::AppendRows2D {
            fill: RowFill::Values {
                added_rows: k,
                data: vec![1.25; k * c],
            },
        },
        2 => TensorEdit::AppendCols2D {
            fill: ColFill::Values {
                added_cols: k,
                data: vec![1.25; r * k],
            },
        },
        3 => TensorEdit::AppendRows2D {
            fill: RowFill::Zeros { added_rows: k },
        },
        _ => TensorEdit::AppendCols2D {
            fill: ColFill::Zeros { added_cols: k },
        },
    };
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit("w", edit).unwrap();
    let receipt = apply_checkpoint_patch(input.path(), &output, &plan).unwrap();
    let bytes = fs::read(output.join("model.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    let t = st.tensor("w").unwrap();
    let width = if dtype == Dtype::F32 { 4 } else { 2 };
    if mode == 0 {
        let vals: Vec<_> = raw
            .chunks_exact(width)
            .map(|b| {
                if dtype == Dtype::F32 {
                    f32::from_le_bytes(b.try_into().unwrap())
                } else if dtype == Dtype::F16 {
                    half::f16::from_le_bytes(b.try_into().unwrap()).to_f32()
                } else {
                    half::bf16::from_le_bytes(b.try_into().unwrap()).to_f32()
                }
            })
            .map(|v| v + 0.25)
            .collect();
        assert_eq!(t.data(), encode(dtype, &vals));
        assert_eq!(receipt.tensor_byte_delta, 0);
    } else if mode == 1 || mode == 3 {
        assert_eq!(&t.data()[..raw.len()], raw);
        assert_eq!(t.shape(), [r + k, c]);
        assert_eq!(
            &t.data()[raw.len()..],
            encode(dtype, &vec![if mode == 1 { 1.25 } else { 0. }; k * c])
        );
    } else {
        assert_eq!(t.shape(), [r, c + k]);
        for row in 0..r {
            let start = row * (c + k) * width;
            assert_eq!(
                &t.data()[start..start + c * width],
                &raw[row * c * width..(row + 1) * c * width]
            );
            assert_eq!(
                &t.data()[start + c * width..start + (c + k) * width],
                encode(dtype, &vec![if mode == 2 { 1.25 } else { 0. }; k])
            );
        }
    }
    assert_eq!(receipt.shards[0].bytes, bytes.len() as u64);
    assert_eq!(
        receipt.shards[0].sha256,
        format!("{:x}", Sha256::digest(&bytes))
    );
    assert_eq!(
        receipt.shards[0].byte_delta,
        receipt.shards[0].bytes as i64 - receipt.shards[0].input_bytes as i64
    );
    assert_eq!(
        SafeTensors::read_metadata(&bytes)
            .unwrap()
            .1
            .metadata()
            .as_ref()
            .unwrap()["origin"],
        "test"
    );
    let output2 = parent.path().join("out2");
    assert_eq!(
        receipt,
        apply_checkpoint_patch(input.path(), &output2, &plan).unwrap()
    );
    assert_eq!(bytes, fs::read(output2.join("model.safetensors")).unwrap());
}
macro_rules! property {($name:ident,$dtype:expr,$mode:expr)=>{proptest!{#![proptest_config(ProptestConfig{cases:256,failure_persistence:None,..ProptestConfig::default()})]
#[test] fn $name(r in 0usize..5,c in 0usize..6,k in 1usize..4,v in -100i16..100){let dtype=$dtype;run(dtype,r,c,k,$mode,encode(dtype,&vec![f32::from(v);r*c]));}}};}
property!(add_f32, Dtype::F32, 0);
property!(add_f16, Dtype::F16, 0);
property!(add_bf16, Dtype::BF16, 0);
property!(rows_f32, Dtype::F32, 1);
property!(rows_f16, Dtype::F16, 1);
property!(rows_bf16, Dtype::BF16, 1);
property!(cols_f32, Dtype::F32, 2);
property!(cols_f16, Dtype::F16, 2);
property!(cols_bf16, Dtype::BF16, 2);
property!(zero_rows_f32, Dtype::F32, 3);
property!(zero_rows_f16, Dtype::F16, 3);
property!(zero_rows_bf16, Dtype::BF16, 3);
property!(zero_cols_f32, Dtype::F32, 4);
property!(zero_cols_f16, Dtype::F16, 4);
property!(zero_cols_bf16, Dtype::BF16, 4);
#[test]
fn exotic_prefixes() {
    for mode in 1..5 {
        run(
            Dtype::BF16,
            1,
            3,
            2,
            mode,
            [0x7fc1u16, 0xff82, 0x8000]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        );
    }
}
#[test]
fn streaming_failure_cleans_stage() {
    let input = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    fixture(
        input.path(),
        Dtype::F16,
        vec![1, 1],
        &encode(Dtype::F16, &[65500.]),
    );
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit(
        "w",
        TensorEdit::AddDense2D {
            rows: 1,
            cols: 1,
            delta_row_major: vec![100.],
        },
    )
    .unwrap();
    assert!(matches!(
        apply_checkpoint_patch(input.path(), &parent.path().join("out"), &plan),
        Err(SurgeonError::NonFiniteResult { row: 0, col: 0, .. })
    ));
    assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
}
#[test]
fn wide_rows() {
    for mode in [0, 2] {
        run(
            Dtype::F32,
            2,
            1_500_000,
            1,
            mode,
            encode(Dtype::F32, &vec![1.; 3_000_000]),
        );
    }
}
#[test]
fn rejection_errors_and_destination_coordinates() {
    let input = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("out");
    fixture(
        input.path(),
        Dtype::F16,
        vec![2, 2],
        &encode(Dtype::F16, &[1.; 4]),
    );
    let bad = [
        (
            TensorEdit::AppendRows2D {
                fill: RowFill::Zeros { added_rows: 0 },
            },
            0,
        ),
        (
            TensorEdit::AppendCols2D {
                fill: ColFill::Values {
                    added_cols: 1,
                    data: vec![70000., 0.],
                },
            },
            1,
        ),
        (
            TensorEdit::AppendRows2D {
                fill: RowFill::Values {
                    added_rows: 1,
                    data: vec![0., f32::NAN],
                },
            },
            2,
        ),
        (
            TensorEdit::AddDense2D {
                rows: 1,
                cols: 4,
                delta_row_major: vec![0.; 4],
            },
            3,
        ),
        (
            TensorEdit::AppendRows2D {
                fill: RowFill::Zeros {
                    added_rows: usize::MAX,
                },
            },
            4,
        ),
    ];
    for (edit, case) in bad {
        let mut plan = CheckpointPatchPlan::default();
        plan.insert_edit("w", edit).unwrap();
        let error = apply_checkpoint_patch(input.path(), &output, &plan).unwrap_err();
        match case {
            0 => assert!(matches!(error, SurgeonError::BufferLengthMismatch { .. })),
            1 => assert!(matches!(
                error,
                SurgeonError::NonFiniteResult { row: 0, col: 2, .. }
            )),
            2 => assert!(matches!(
                error,
                SurgeonError::NonFiniteResult { row: 2, col: 1, .. }
            )),
            3 => assert!(matches!(error, SurgeonError::DimensionMismatch { .. })),
            _ => assert!(matches!(error, SurgeonError::SafetensorsFormat { .. })),
        }
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }
}
#[test]
fn header_declaration_order_and_untouched_integer() {
    let input = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let header=br#"{"z":{"dtype":"F32","shape":[1,1],"data_offsets":[0,4]},"a":{"dtype":"I64","shape":[1],"data_offsets":[4,12]}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&2f32.to_le_bytes());
    bytes.extend_from_slice(&(-7i64).to_le_bytes());
    fs::write(input.path().join("model.safetensors"), bytes).unwrap();
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit(
        "z",
        TensorEdit::AddDense2D {
            rows: 1,
            cols: 1,
            delta_row_major: vec![1.],
        },
    )
    .unwrap();
    let output = parent.path().join("out");
    apply_checkpoint_patch(input.path(), &output, &plan).unwrap();
    let bytes = fs::read(output.join("model.safetensors")).unwrap();
    let st = SafeTensors::deserialize(&bytes).unwrap();
    assert_eq!(st.tensor("a").unwrap().data(), (-7i64).to_le_bytes());
    assert_eq!(st.tensor("z").unwrap().data(), 3f32.to_le_bytes());
}
#[test]
fn finite_and_rank_validation() {
    let input = tempfile::tempdir().unwrap();
    let parent = tempfile::tempdir().unwrap();
    let output = parent.path().join("out");
    for (dtype, shape, raw) in [
        (Dtype::I64, vec![1, 1], vec![0; 8]),
        (Dtype::F32, vec![1], vec![0; 4]),
        (Dtype::F32, vec![1, 1, 1], vec![0; 4]),
    ] {
        fixture(input.path(), dtype, shape, &raw);
        let mut p = CheckpointPatchPlan::default();
        p.insert_edit(
            "w",
            TensorEdit::AppendRows2D {
                fill: RowFill::Zeros { added_rows: 1 },
            },
        )
        .unwrap();
        let e = apply_checkpoint_patch(input.path(), &output, &p).unwrap_err();
        assert!(matches!(
            e,
            SurgeonError::NotMatrix2D { .. } | SurgeonError::UnsupportedDtype { .. }
        ));
    }
    for (original, delta) in [
        (f32::MAX, f32::MAX),
        (f32::INFINITY, 0.),
        (f32::NAN, 0.),
        (0., f32::NAN),
    ] {
        fixture(
            input.path(),
            Dtype::F32,
            vec![1, 1],
            &original.to_le_bytes(),
        );
        let mut p = CheckpointPatchPlan::default();
        p.insert_edit(
            "w",
            TensorEdit::AddDense2D {
                rows: 1,
                cols: 1,
                delta_row_major: vec![delta],
            },
        )
        .unwrap();
        assert!(matches!(
            apply_checkpoint_patch(input.path(), &output, &p),
            Err(SurgeonError::NonFiniteResult { row: 0, col: 0, .. })
        ));
        assert_eq!(fs::read_dir(parent.path()).unwrap().count(), 0);
    }
}
#[test]
fn rounding_ties() {
    for dtype in [Dtype::BF16, Dtype::F16] {
        let values = if dtype == Dtype::BF16 {
            vec![1. + 1. / 256., 1. + 3. / 256., -1. - 1. / 256.]
        } else {
            vec![1. + 1. / 2048., 1. + 3. / 2048., -1. - 1. / 2048.]
        };
        let input = tempfile::tempdir().unwrap();
        let parent = tempfile::tempdir().unwrap();
        fixture(input.path(), dtype, vec![1, 3], &encode(dtype, &[0.; 3]));
        let mut p = CheckpointPatchPlan::default();
        p.insert_edit(
            "w",
            TensorEdit::AddDense2D {
                rows: 1,
                cols: 3,
                delta_row_major: values.clone(),
            },
        )
        .unwrap();
        let out = parent.path().join("out");
        apply_checkpoint_patch(input.path(), &out, &p).unwrap();
        let bytes = fs::read(out.join("model.safetensors")).unwrap();
        assert_eq!(
            SafeTensors::deserialize(&bytes)
                .unwrap()
                .tensor("w")
                .unwrap()
                .data(),
            encode(dtype, &values)
        );
    }
}
