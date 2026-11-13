use proptest::prelude::*;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use safetensors_surgeon::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

type Tensor = (String, Dtype, Vec<usize>, Vec<u8>);

fn dirs() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("input");
    let output = root.path().join("output");
    fs::create_dir(&input).unwrap();
    (root, input, output)
}

fn matrix(name: &str, value: f32) -> Tensor {
    (
        name.into(),
        Dtype::F32,
        vec![1, 1],
        value.to_le_bytes().to_vec(),
    )
}

fn shard(
    dir: &Path,
    filename: &str,
    tensors: &[Tensor],
    metadata: Option<HashMap<String, String>>,
) {
    let views = tensors.iter().map(|(name, dtype, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        )
    });
    fs::write(
        dir.join(filename),
        safetensors::serialize(views, metadata).unwrap(),
    )
    .unwrap();
}

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn index(input: &Path, weights: Value) {
    write_json(
        &input.join("model.safetensors.index.json"),
        &json!({
            "metadata": {"total_size": 999999, "origin": "fixture"},
            "weight_map": weights,
            "custom": {"keep": true}
        }),
    );
}

fn append(name: &str) -> CheckpointPatchPlan {
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit(
        name,
        TensorEdit::AppendRows2D {
            fill: RowFill::Zeros { added_rows: 1 },
        },
    )
    .unwrap();
    plan
}

fn assert_clean(output: &Path) {
    if output.exists() {
        assert_eq!(fs::read_dir(output).unwrap().count(), 0);
    }
    for entry in fs::read_dir(output.parent().unwrap()).unwrap() {
        assert!(!entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".safetensors-surgeon"));
    }
}

#[test]
fn sharded_receipts_auxiliaries_and_untouched_strategies() {
    for strategy in [
        OutputStrategy::CopyUnmodifiedShards,
        OutputStrategy::HardlinkUnmodifiedShards,
    ] {
        let (_root, input, output) = dirs();
        shard(&input, "a.safetensors", &[matrix("a", 1.)], None);
        shard(&input, "b.safetensors", &[matrix("b", 2.)], None);
        index(&input, json!({"a":"a.safetensors","b":"b.safetensors"}));
        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ] {
            fs::write(input.join(filename), b"{\n  \"untouched\": true\n}\n").unwrap();
        }
        let mut plan = append("a");
        plan.output_strategy = strategy;
        let receipt = apply_checkpoint_patch(&input, &output, &plan).unwrap();
        assert!(receipt.sharded);
        assert_eq!(receipt.total_input_tensor_bytes, 8);
        assert_eq!(receipt.total_tensor_bytes, 12);
        assert_eq!(receipt.tensor_byte_delta, 4);
        assert_eq!(receipt.tensors_modified, 1);
        assert_eq!(
            receipt
                .shards
                .iter()
                .map(|s| s.filename.as_str())
                .collect::<Vec<_>>(),
            ["a.safetensors", "b.safetensors"]
        );
        for result in &receipt.shards {
            let bytes = fs::read(output.join(&result.filename)).unwrap();
            SafeTensors::deserialize(&bytes).unwrap();
            assert_eq!(result.bytes, bytes.len() as u64);
            assert_eq!(result.sha256, format!("{:x}", Sha256::digest(&bytes)));
            assert_eq!(
                result.byte_delta,
                result.bytes as i64 - result.input_bytes as i64
            );
        }
        let untouched = &receipt.shards[1];
        assert!(!untouched.modified);
        assert_eq!(
            untouched.hardlinked,
            strategy == OutputStrategy::HardlinkUnmodifiedShards
        );
        assert_eq!(
            fs::read(input.join("b.safetensors")).unwrap(),
            fs::read(output.join("b.safetensors")).unwrap()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let before = fs::metadata(input.join("b.safetensors")).unwrap();
            let after = fs::metadata(output.join("b.safetensors")).unwrap();
            assert_eq!(
                (before.dev(), before.ino()) == (after.dev(), after.ino()),
                untouched.hardlinked
            );
        }
        let result: Value =
            serde_json::from_slice(&fs::read(output.join("model.safetensors.index.json")).unwrap())
                .unwrap();
        assert_eq!(result["metadata"]["total_size"], 12);
        assert_eq!(result["metadata"]["origin"], "fixture");
        assert_eq!(result["custom"], json!({"keep":true}));
        for filename in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
        ] {
            assert_eq!(
                fs::read(input.join(filename)).unwrap(),
                fs::read(output.join(filename)).unwrap()
            );
        }
    }
}

#[test]
fn index_is_authoritative_and_metadata_is_optional() {
    let (_root, input, output) = dirs();
    shard(&input, "a.safetensors", &[matrix("a", 1.)], None);
    fs::write(input.join("model.safetensors"), b"ignored invalid file").unwrap();
    write_json(
        &input.join("model.safetensors.index.json"),
        &json!({"weight_map":{"a":"a.safetensors"}}),
    );
    apply_checkpoint_patch(&input, &output, &append("a")).unwrap();
    assert!(!output.join("model.safetensors").exists());
    let result: Value =
        serde_json::from_slice(&fs::read(output.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    assert_eq!(result["metadata"]["total_size"], 8);

    let (_root, input, output) = dirs();
    shard(&input, "model.safetensors", &[matrix("a", 1.)], None);
    fs::write(input.join("model.safetensors.index.json"), b"invalid").unwrap();
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &append("a")),
        Err(SurgeonError::Json { .. })
    ));
    assert!(!output.exists());
    assert_clean(&output);
}

#[test]
fn index_requires_exact_tensor_ownership() {
    for weights in [
        json!({"a":"b.safetensors","b":"a.safetensors"}),
        json!({"absent":"a.safetensors","b":"b.safetensors"}),
        json!({"a":"a.safetensors","missing":"b.safetensors"}),
    ] {
        let (_root, input, output) = dirs();
        shard(&input, "a.safetensors", &[matrix("a", 1.)], None);
        shard(&input, "b.safetensors", &[matrix("b", 2.)], None);
        index(&input, weights);
        assert!(matches!(
            apply_checkpoint_patch(&input, &output, &CheckpointPatchPlan::default()),
            Err(SurgeonError::SafetensorsFormat { .. })
        ));
        assert_clean(&output);
    }
    for duplicate in [false, true] {
        let (_root, input, output) = dirs();
        shard(
            &input,
            "a.safetensors",
            &[matrix("a", 1.), matrix("extra", 2.)],
            None,
        );
        shard(
            &input,
            "b.safetensors",
            &[matrix(if duplicate { "extra" } else { "b" }, 3.)],
            None,
        );
        index(&input, json!({"a":"a.safetensors","b":"b.safetensors"}));
        assert!(matches!(
            apply_checkpoint_patch(&input, &output, &append("a")),
            Err(SurgeonError::SafetensorsFormat { .. })
        ));
        assert!(!output.exists());
        assert_clean(&output);
    }
}

#[test]
fn malformed_index_structure_rejected() {
    for value in [
        json!([]),
        json!({}),
        json!({"weight_map":[]}),
        json!({"weight_map":{"a":42}}),
        json!({"weight_map":{},"metadata":[]}),
    ] {
        let (_root, input, output) = dirs();
        write_json(&input.join("model.safetensors.index.json"), &value);
        assert!(matches!(
            apply_checkpoint_patch(&input, &output, &CheckpointPatchPlan::default()),
            Err(SurgeonError::SafetensorsFormat { .. })
        ));
        assert_clean(&output);
    }
}

#[test]
fn unsafe_and_colliding_shard_names_rejected_before_writing() {
    for name in [
        "../outside.safetensors",
        "/tmp/outside.safetensors",
        "sub/file.safetensors",
        "sub/../file.safetensors",
        "..",
        ".",
        "",
        "C:\\outside.safetensors",
        "a\\b.safetensors",
        "foo..safetensors",
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ] {
        let (_root, input, output) = dirs();
        index(&input, json!({"w": name}));
        assert!(
            matches!(apply_checkpoint_patch(&input, &output, &CheckpointPatchPlan::default()), Err(SurgeonError::UnsafeShardFilename(found)) if found == name)
        );
        assert!(!output.exists());
        assert_clean(&output);
    }
}

#[cfg(unix)]
#[test]
fn cache_symlink_shards_work_for_copy_and_hardlink() {
    use std::os::unix::fs::{symlink, MetadataExt};
    for strategy in [
        OutputStrategy::CopyUnmodifiedShards,
        OutputStrategy::HardlinkUnmodifiedShards,
    ] {
        let (root, input, output) = dirs();
        let blobs = root.path().join("blobs");
        fs::create_dir(&blobs).unwrap();
        shard(&blobs, "blob", &[matrix("w", 1.)], None);
        symlink("../blobs/blob", input.join("cached.safetensors")).unwrap();
        index(&input, json!({"w":"cached.safetensors"}));
        let plan = CheckpointPatchPlan {
            output_strategy: strategy,
            ..CheckpointPatchPlan::default()
        };
        let receipt = apply_checkpoint_patch(&input, &output, &plan).unwrap();
        assert!(!fs::symlink_metadata(output.join("cached.safetensors"))
            .unwrap()
            .file_type()
            .is_symlink());
        let before = fs::metadata(blobs.join("blob")).unwrap();
        let after = fs::metadata(output.join("cached.safetensors")).unwrap();
        assert_eq!(
            (before.dev(), before.ino()) == (after.dev(), after.ino()),
            receipt.shards[0].hardlinked
        );
        assert_eq!(
            receipt.shards[0].hardlinked,
            strategy == OutputStrategy::HardlinkUnmodifiedShards
        );
    }
}

fn config_plan() -> CheckpointPatchPlan {
    CheckpointPatchPlan {
        config_intermediate_size: Some(ConfigIntermediateSizeUpdate {
            expected_old_size: 4,
            new_size: 7,
        }),
        ..CheckpointPatchPlan::default()
    }
}

#[test]
fn config_updates_every_present_size_preserving_other_content() {
    for mut config in [
        json!({"intermediate_size":4,"other":[1,"x"]}),
        json!({"text_config":{"intermediate_size":4,"other":true},"other":[1,"x"]}),
        json!({"intermediate_size":4,"text_config":{"intermediate_size":4,"other":true},"other":[1,"x"]}),
    ] {
        let (_root, input, output) = dirs();
        shard(&input, "model.safetensors", &[matrix("w", 1.)], None);
        write_json(&input.join("config.json"), &config);
        apply_checkpoint_patch(&input, &output, &config_plan()).unwrap();
        if config.get("intermediate_size").is_some() {
            config["intermediate_size"] = 7.into();
        }
        if config.pointer("/text_config/intermediate_size").is_some() {
            config["text_config"]["intermediate_size"] = 7.into();
        }
        let actual: Value =
            serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
        assert_eq!(actual, config);
    }
}

#[test]
fn invalid_config_updates_fail_before_output() {
    for config in [
        json!({}),
        json!([]),
        json!({"intermediate_size":3}),
        json!({"intermediate_size":-4}),
        json!({"intermediate_size":4.0}),
        json!({"intermediate_size":"4"}),
        json!({"text_config":{}}),
        json!({"intermediate_size":4,"text_config":{"intermediate_size":5}}),
        json!({"intermediate_size":3,"text_config":{"intermediate_size":4}}),
        json!({"intermediate_size":4,"text_config":null}),
    ] {
        let (_root, input, output) = dirs();
        shard(&input, "model.safetensors", &[matrix("w", 1.)], None);
        write_json(&input.join("config.json"), &config);
        assert!(matches!(
            apply_checkpoint_patch(&input, &output, &config_plan()),
            Err(SurgeonError::ConfigMismatch(_))
        ));
        assert!(!output.exists());
        assert_clean(&output);
    }
    let (_root, input, output) = dirs();
    shard(&input, "model.safetensors", &[matrix("w", 1.)], None);
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &config_plan()),
        Err(SurgeonError::ConfigMismatch(_))
    ));
    assert_clean(&output);
}

#[test]
fn shape_assertions_and_missing_targets_fail_closed() {
    let (_root, input, output) = dirs();
    shard(&input, "model.safetensors", &[matrix("w", 1.)], None);
    let mut plan = append("w");
    plan.expected_shapes.insert("w".into(), (2, 1));
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &plan),
        Err(SurgeonError::DimensionMismatch {
            expected: (2, 1),
            actual: (1, 1),
            ..
        })
    ));
    assert_clean(&output);
    plan.expected_shapes.insert("w".into(), (1, 1));
    apply_checkpoint_patch(&input, &output, &plan).unwrap();
    let missing_output = output.parent().unwrap().join("missing");
    plan.expected_shapes.insert("missing".into(), (1, 1));
    assert!(
        matches!(apply_checkpoint_patch(&input, &missing_output, &plan), Err(SurgeonError::MissingTensor { name }) if name == "missing")
    );
    assert!(
        matches!(apply_checkpoint_patch(&input, &missing_output, &append("absent")), Err(SurgeonError::MissingTensor { name }) if name == "absent")
    );
    assert!(!missing_output.exists());
}

#[test]
fn destination_must_be_absent_or_empty_and_separate() {
    let (root, input, output) = dirs();
    shard(&input, "model.safetensors", &[matrix("w", 1.)], None);
    fs::create_dir(&output).unwrap();
    apply_checkpoint_patch(&input, &output, &append("w")).unwrap();
    let before = fs::read(output.join("model.safetensors")).unwrap();
    assert!(apply_checkpoint_patch(&input, &output, &append("w")).is_err());
    assert_eq!(fs::read(output.join("model.safetensors")).unwrap(), before);
    for invalid in [
        input.clone(),
        input.join("child"),
        root.path().to_path_buf(),
        root.path().join("missing/child"),
    ] {
        assert!(apply_checkpoint_patch(&input, &invalid, &append("w")).is_err());
    }
    assert!(!input.join("child").exists());
    assert!(!root.path().join("missing").exists());
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(&input, root.path().join("alias")).unwrap();
        assert!(apply_checkpoint_patch(&input, &root.path().join("alias"), &append("w")).is_err());
    }
}

#[test]
fn corrupt_untouched_shard_is_validated_before_staging() {
    let (_root, input, output) = dirs();
    shard(&input, "a.safetensors", &[matrix("a", 1.)], None);
    fs::write(input.join("z.safetensors"), b"invalid").unwrap();
    index(&input, json!({"a":"a.safetensors","z":"z.safetensors"}));
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &append("a")),
        Err(SurgeonError::SafetensorsFormat { .. })
    ));
    assert!(!output.exists());
    assert_clean(&output);
}

#[test]
fn malformed_headers_and_trailing_payload_rejected() {
    let valid = safetensors::serialize(
        [(
            "w",
            TensorView::new(Dtype::F32, vec![1, 1], &1f32.to_le_bytes()).unwrap(),
        )],
        None,
    )
    .unwrap();
    let raw = |header: &[u8], data: &[u8]| {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(data);
        bytes
    };
    let mut trailing = valid.clone();
    trailing.push(0);
    let mut truncated = valid;
    truncated.pop();
    for bad in [
        vec![],
        vec![0; 7],
        100_000_001u64.to_le_bytes().to_vec(),
        raw(&[0xff], &[]),
        raw(b"not json", &[]),
        raw(
            br#"{"w":{"dtype":"F32","shape":[1,1],"data_offsets":[1,5]}}"#,
            &[0; 5],
        ),
        raw(
            br#"{"w":{"dtype":"F32","shape":[1,2],"data_offsets":[0,4]}}"#,
            &[0; 4],
        ),
        trailing,
        truncated,
    ] {
        let (_root, input, output) = dirs();
        fs::write(input.join("model.safetensors"), bad).unwrap();
        assert!(matches!(
            apply_checkpoint_patch(&input, &output, &CheckpointPatchPlan::default()),
            Err(SurgeonError::SafetensorsFormat { .. })
        ));
        assert!(!output.exists());
        assert_clean(&output);
    }
}

#[test]
fn late_multishard_overflow_rolls_back_absent_and_empty_outputs() {
    for existing in [false, true] {
        let (_root, input, output) = dirs();
        shard(&input, "a.safetensors", &[matrix("a", 1.)], None);
        let half = (
            "b".into(),
            Dtype::F16,
            vec![1, 1],
            half::f16::from_f32(65_500.).to_le_bytes().to_vec(),
        );
        shard(&input, "b.safetensors", &[half], None);
        shard(&input, "c.safetensors", &[matrix("c", 2.)], None);
        index(
            &input,
            json!({"a":"a.safetensors","b":"b.safetensors","c":"c.safetensors"}),
        );
        let original = fs::read(input.join("a.safetensors")).unwrap();
        let mut plan = append("a");
        plan.insert_edit(
            "b",
            TensorEdit::AddDense2D {
                rows: 1,
                cols: 1,
                delta_row_major: vec![100.],
            },
        )
        .unwrap();
        if existing {
            fs::create_dir(&output).unwrap();
        }
        assert!(
            matches!(apply_checkpoint_patch(&input, &output, &plan), Err(SurgeonError::NonFiniteResult { name, row:0, col:0 }) if name == "b")
        );
        assert_eq!(output.exists(), existing);
        assert_clean(&output);
        assert_eq!(fs::read(input.join("a.safetensors")).unwrap(), original);
    }
}

fn float_dtype(value: u8) -> Dtype {
    [Dtype::BF16, Dtype::F16, Dtype::F32][value as usize]
}

proptest! {
    #![proptest_config(ProptestConfig { cases:256, failure_persistence:None, ..ProptestConfig::default() })]
    #[test]
    fn random_shard_splits_preserve_bytes_and_exact_accounting(
        specs in prop::collection::vec((0u8..4, 0u8..3, 0usize..5, 0usize..5, any::<u32>()), 1..9),
        added in 1usize..4,
    ) {
        let (_root, input, output) = dirs();
        let mut groups: BTreeMap<u8, Vec<Tensor>> = BTreeMap::new();
        let mut weights = serde_json::Map::new();
        let mut originals = BTreeMap::new();
        let mut input_bytes = 0u64;
        for (i, &(bucket, dtype, rows, cols, bits)) in specs.iter().enumerate() {
            let name = format!("tensor_{i}");
            let dtype = float_dtype(dtype);
            let width = if dtype == Dtype::F32 {4} else {2};
            let data:Vec<u8> = bits.to_le_bytes()[..width].iter().copied().cycle().take(rows*cols*width).collect();
            input_bytes += data.len() as u64;
            originals.insert(name.clone(), (dtype, rows, cols, data.clone()));
            groups.entry(bucket).or_default().push((name.clone(),dtype,vec![rows,cols],data));
            weights.insert(name, format!("part-{bucket}.safetensors").into());
        }
        for (bucket,tensors) in &groups { shard(&input,&format!("part-{bucket}.safetensors"),tensors,None); }
        index(&input, weights.clone().into());
        let mut plan=CheckpointPatchPlan::default();
        plan.insert_edit("tensor_0",TensorEdit::AppendCols2D { fill:ColFill::Zeros { added_cols:added } }).unwrap();
        let receipt=apply_checkpoint_patch(&input,&output,&plan).unwrap();
        let first=&originals["tensor_0"];
        let delta=first.1*added*if first.0==Dtype::F32 {4}else{2};
        prop_assert_eq!(receipt.total_input_tensor_bytes,input_bytes);
        prop_assert_eq!(receipt.total_tensor_bytes,input_bytes+delta as u64);
        for (name,(dtype,rows,cols,raw)) in &originals {
            let bytes=fs::read(output.join(weights[name].as_str().unwrap())).unwrap();
            let parsed=SafeTensors::deserialize(&bytes).unwrap();
            let actual=parsed.tensor(name).unwrap();
            if name=="tensor_0" {
                let width=if *dtype==Dtype::F32 {4}else{2};
                prop_assert_eq!(actual.shape(),&[*rows,*cols+added]);
                for row in 0..*rows {
                    let start=row*(*cols+added)*width;
                    prop_assert_eq!(&actual.data()[start..start+*cols*width],&raw[row*cols*width..(row+1)*cols*width]);
                    prop_assert!(actual.data()[start+cols*width..start+(cols+added)*width].iter().all(|b|*b==0));
                }
            } else {prop_assert_eq!(actual.data(),raw.as_slice());}
        }
        let updated:Value=serde_json::from_slice(&fs::read(output.join("model.safetensors.index.json")).unwrap()).unwrap();
        prop_assert_eq!(&updated["weight_map"],&Value::Object(weights));
        prop_assert_eq!(updated["metadata"]["total_size"].as_u64(),Some(receipt.total_tensor_bytes));
    }

    #[test]
    fn arbitrary_metadata_survives_rewriting(
        metadata in prop::collection::btree_map("[a-zA-Z_é中]{0,8}", ".{0,16}",0..8),
        present in any::<bool>(), dtype in 0u8..3,
    ) {
        let (_root,input,output)=dirs();
        let metadata=present.then(||metadata.into_iter().collect::<HashMap<_,_>>());
        let dtype=float_dtype(dtype);
        let bytes=if dtype==Dtype::F32 {vec![0;4]}else{vec![0;2]};
        shard(&input,"model.safetensors",&[("w".into(),dtype,vec![1,1],bytes)],metadata.clone());
        apply_checkpoint_patch(&input,&output,&append("w")).unwrap();
        let bytes=fs::read(output.join("model.safetensors")).unwrap();
        SafeTensors::deserialize(&bytes).unwrap();
        let parsed_metadata=SafeTensors::read_metadata(&bytes).unwrap().1;
        prop_assert_eq!(parsed_metadata.metadata(),&metadata);
    }

    #[test]
    fn random_mixed_layouts_preserve_all_unedited_tensors(
        specs in prop::collection::vec((0u8..5,0usize..8,any::<u64>()),0..9),
    ) {
        let (_root,input,output)=dirs();
        let mut tensors=vec![matrix("edited",1.)];
        for (i,(kind,count,bits)) in specs.iter().enumerate() {
            let dtype=[Dtype::U8,Dtype::I64,Dtype::F16,Dtype::BF16,Dtype::F32][*kind as usize];
            let width=match dtype {Dtype::U8=>1,Dtype::I64=>8,Dtype::F32=>4,_=>2};
            let raw=bits.to_le_bytes()[..width].iter().copied().cycle().take(count*width).collect();
            tensors.push((format!("other_{i}"),dtype,vec![*count],raw));
        }
        shard(&input,"model.safetensors",&tensors,None);
        apply_checkpoint_patch(&input,&output,&append("edited")).unwrap();
        let bytes=fs::read(output.join("model.safetensors")).unwrap();
        let parsed=SafeTensors::deserialize(&bytes).unwrap();
        for (name,dtype,shape,data) in tensors.iter().skip(1) {
            let actual=parsed.tensor(name).unwrap();
            prop_assert_eq!(actual.dtype(),*dtype);
            prop_assert_eq!(actual.shape(),shape.as_slice());
            prop_assert_eq!(actual.data(),data.as_slice());
        }
        let (_,metadata)=SafeTensors::read_metadata(&bytes).unwrap();
        let mut cursor=0;
        for name in metadata.offset_keys() {
            let info=metadata.info(&name).unwrap();
            prop_assert_eq!(info.data_offsets.0,cursor);
            cursor=info.data_offsets.1;
        }
        prop_assert_eq!(cursor,metadata.data_len());
    }
}

#[test]
fn lowered_mlp_validates_zero_filled_layers_and_widens_all_projections() {
    let (_root, input, output) = dirs();
    let mut tensors = Vec::new();
    for layer in 0..2 {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            let shape = if projection == "down_proj" {
                vec![32, 64]
            } else {
                vec![64, 32]
            };
            tensors.push((
                format!("model.layers.{layer}.mlp.{projection}.weight"),
                Dtype::F32,
                shape,
                vec![0; 64 * 32 * 4],
            ));
        }
    }
    let plan = MlpWidenPlan {
        layer_prefix: "model.layers".into(),
        num_layers: 2,
        hidden_size: 32,
        base_intermediate_size: 64,
        added_slots: 2,
        target_layers: vec![LayerMlpAppendSpec {
            layer_index: 0,
            gate_rows: vec![1.; 64],
            up_rows: vec![2.; 64],
            down_cols: vec![3.; 64],
        }],
        uniform_across_all_layers: true,
        output_strategy: OutputStrategy::default(),
    }
    .into_patch_plan()
    .unwrap();
    write_json(&input.join("config.json"), &json!({"intermediate_size":64}));
    tensors[3].2 = vec![64, 48];
    tensors[3].3 = vec![0; 64 * 48 * 4];
    shard(&input, "model.safetensors", &tensors, None);
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &plan),
        Err(SurgeonError::DimensionMismatch {
            expected: (64, 32),
            actual: (64, 48),
            ..
        })
    ));
    assert_clean(&output);
    tensors[3].2 = vec![64, 32];
    tensors[3].3 = vec![0; 64 * 32 * 4];
    shard(&input, "model.safetensors", &tensors, None);
    let receipt = apply_checkpoint_patch(&input, &output, &plan).unwrap();
    assert_eq!(receipt.tensors_modified, 6);
    let bytes = fs::read(output.join("model.safetensors")).unwrap();
    let parsed = SafeTensors::deserialize(&bytes).unwrap();
    for (name, _, _, _) in tensors {
        let tensor = parsed.tensor(&name).unwrap();
        assert_eq!(
            tensor.shape(),
            if name.contains("down_proj") {
                [32, 66]
            } else {
                [66, 32]
            }
        );
        if name.contains("layers.1") {
            assert!(tensor.data().iter().all(|b| *b == 0));
        }
    }
    let config: Value =
        serde_json::from_slice(&fs::read(output.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["intermediate_size"], 66);
}

#[test]
fn official_bit_count_overflow_is_rejected_before_staging() {
    let (_root, input, output) = dirs();
    shard(
        &input,
        "model.safetensors",
        &[("w".into(), Dtype::F32, vec![0, 1], vec![])],
        None,
    );
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit(
        "w",
        TensorEdit::AppendRows2D {
            fill: RowFill::Zeros {
                added_rows: usize::MAX / 32 + 1,
            },
        },
    )
    .unwrap();
    assert!(matches!(
        apply_checkpoint_patch(&input, &output, &plan),
        Err(SurgeonError::SafetensorsFormat { .. })
    ));
    assert!(!output.exists());
    assert_clean(&output);
}
