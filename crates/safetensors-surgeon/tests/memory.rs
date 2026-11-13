#![cfg(target_os = "linux")]

use safetensors_surgeon::{apply_checkpoint_patch, CheckpointPatchPlan, ColFill, TensorEdit};
use serde_json::json;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::process::Command;

const SHARD_BYTES: u64 = 64 * 1024 * 1024;
const CHILD_FLAG: &str = "SAFETENSORS_SURGEON_RSS_CHILD";

#[test]
fn rss_is_bounded_in_subprocess() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "rss_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_FLAG, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "RSS subprocess failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    print!("{}", String::from_utf8_lossy(&output.stdout));
}

#[test]
fn rss_child() {
    if std::env::var_os(CHILD_FLAG).is_none() {
        return;
    }
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("input");
    let output = directory.path().join("output");
    fs::create_dir(&input).unwrap();
    create_sparse_checkpoint(&input);
    let mut plan = CheckpointPatchPlan::default();
    plan.insert_edit(
        "weight0",
        TensorEdit::AppendCols2D {
            fill: ColFill::Zeros { added_cols: 1 },
        },
    )
    .unwrap();
    let receipt = apply_checkpoint_patch(&input, &output, &plan).unwrap();
    assert_eq!(receipt.total_input_tensor_bytes, 4 * SHARD_BYTES);
    assert_eq!(receipt.total_tensor_bytes, 4 * SHARD_BYTES + 4);
    assert_eq!(receipt.tensor_byte_delta, 4);
    assert_eq!(receipt.shards.len(), 4);
    assert_eq!(receipt.tensors_modified, 1);
    let mut file = File::open(output.join("model-00001-of-00004.safetensors")).unwrap();
    let mut prefix = [0_u8; 8];
    file.read_exact(&mut prefix).unwrap();
    let mut header = vec![0; u64::from_le_bytes(prefix) as usize];
    file.read_exact(&mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();
    assert_eq!(header["weight0"]["shape"], json!([1, SHARD_BYTES / 4 + 1]));

    let status = fs::read_to_string("/proc/self/status").unwrap();
    let high_water_kib = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .parse::<u64>()
        .unwrap();
    println!("peak RSS: {high_water_kib} KiB (limit: 65536 KiB)");
    assert!(
        high_water_kib <= 64 * 1024,
        "peak RSS was {high_water_kib} KiB; limit is 65536 KiB"
    );
}

fn create_sparse_checkpoint(input: &Path) {
    let mut weight_map = serde_json::Map::new();
    for shard in 0..4 {
        let name = format!("weight{shard}");
        let filename = format!("model-{:05}-of-00004.safetensors", shard + 1);
        let mut tensors = serde_json::Map::new();
        tensors.insert(
            name.clone(),
            json!({
                "dtype": "F32",
                "shape": [1, SHARD_BYTES / 4],
                "data_offsets": [0, SHARD_BYTES]
            }),
        );
        let mut header = serde_json::to_vec(&tensors).unwrap();
        header.resize(header.len().div_ceil(8) * 8, b' ');
        let mut file = File::create(input.join(&filename)).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.set_len(8 + header.len() as u64 + SHARD_BYTES).unwrap();
        weight_map.insert(name, json!(filename));
    }
    serde_json::to_writer(
        File::create(input.join("model.safetensors.index.json")).unwrap(),
        &json!({ "metadata": { "total_size": 4 * SHARD_BYTES }, "weight_map": weight_map }),
    )
    .unwrap();
}
