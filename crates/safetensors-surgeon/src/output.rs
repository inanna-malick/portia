use crate::checkpoint::{PlannedRewrite, Preflight, Shard};
use crate::{
    mmap, tensor, CheckpointPatchPlan, OutputStrategy, PatchReceipt, ShardReceipt, SurgeonError,
};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

fn io_error(path: &Path, source: io::Error) -> SurgeonError {
    SurgeonError::Io {
        path: path.into(),
        source,
    }
}
fn format_error(path: &Path) -> SurgeonError {
    SurgeonError::SafetensorsFormat {
        path: path.into(),
        message: "byte accounting overflow".into(),
    }
}
fn delta(path: &Path, after: u64, before: u64) -> Result<i64, SurgeonError> {
    i64::try_from(i128::from(after) - i128::from(before)).map_err(|_| format_error(path))
}
struct HashWriter<W> {
    inner: W,
    hash: Sha256,
}
impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}
fn hash_file(path: &Path) -> Result<String, SurgeonError> {
    let file = File::open(path).map_err(|e| io_error(path, e))?;
    let len = file.metadata().map_err(|e| io_error(path, e))?.len();
    let mut reader = mmap::Reader::new(&file, path)?;
    let mut sink = HashWriter {
        inner: io::sink(),
        hash: Sha256::new(),
    };
    reader.copy_range(
        0,
        usize::try_from(len).map_err(|_| format_error(path))?,
        &mut sink,
    )?;
    Ok(format!("{:x}", sink.hash.finalize()))
}
fn rewrite(
    shard: &Shard,
    planned: &PlannedRewrite,
    dest: &Path,
    plan: &CheckpointPatchPlan,
) -> Result<String, SurgeonError> {
    let bytes = &planned.header_bytes;
    let temporary = tempfile::NamedTempFile::new_in(dest.parent().expect("staging directory"))
        .map_err(|e| io_error(dest, e))?;
    let file = temporary.reopen().map_err(|e| io_error(dest, e))?;
    let mut writer = HashWriter {
        inner: BufWriter::with_capacity(1024 * 1024, file),
        hash: Sha256::new(),
    };
    writer
        .write_all(&(bytes.len() as u64).to_le_bytes())
        .map_err(|e| io_error(dest, e))?;
    writer.write_all(bytes).map_err(|e| io_error(dest, e))?;
    let input = File::open(&shard.path).map_err(|e| io_error(&shard.path, e))?;
    let mut reader = mmap::Reader::new(&input, &shard.path)?;
    let mut buffer = vec![0; tensor::BUFFER_BYTES];
    for (name, info) in &shard.tensors {
        match plan.edits.get(name) {
            Some(edit) => tensor::write(
                name,
                info,
                &mut reader,
                shard.data_start,
                edit,
                &mut buffer,
                &mut writer,
            )?,
            None => reader.copy_range(
                shard.data_start + info.data_offsets.0 as u64,
                info.data_offsets.1 - info.data_offsets.0,
                &mut writer,
            )?,
        }
    }
    writer.flush().map_err(|e| io_error(dest, e))?;
    writer
        .inner
        .get_ref()
        .sync_all()
        .map_err(|e| io_error(dest, e))?;
    let hash = format!("{:x}", writer.hash.finalize());
    drop(writer.inner);
    temporary
        .persist(dest)
        .map_err(|e| io_error(dest, e.error))?;
    Ok(hash)
}
fn link_or_copy(source: &Path, dest: &Path, hardlink: bool) -> Result<bool, SurgeonError> {
    link_with(source, dest, hardlink, |a, b| fs::hard_link(a, b))
}
fn link_with(
    source: &Path,
    dest: &Path,
    hardlink: bool,
    hard_link: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> Result<bool, SurgeonError> {
    if hardlink {
        let source = fs::canonicalize(source).map_err(|e| io_error(source, e))?;
        match hard_link(&source, dest) {
            Ok(()) => return Ok(true),
            Err(e) if e.kind() == io::ErrorKind::CrossesDevices => {}
            Err(e) => return Err(io_error(dest, e)),
        }
    }
    fs::copy(source, dest).map_err(|e| io_error(dest, e))?;
    Ok(false)
}
fn json_file(path: &Path, value: &serde_json::Value) -> Result<(), SurgeonError> {
    let mut file = File::create(path).map_err(|e| io_error(path, e))?;
    serde_json::to_writer(&mut file, value).map_err(|source| SurgeonError::Json {
        path: path.into(),
        source,
    })?;
    file.sync_all().map_err(|e| io_error(path, e))
}
pub(crate) fn apply(
    mut pre: Preflight,
    plan: &CheckpointPatchPlan,
) -> Result<PatchReceipt, SurgeonError> {
    let parent = pre.output_dir.parent().unwrap_or(Path::new("."));
    let stage = tempfile::Builder::new()
        .prefix(".safetensors-surgeon-stage-")
        .tempdir_in(parent)
        .map_err(|e| io_error(parent, e))?;
    let mut receipts = Vec::new();
    let mut total = 0u64;
    for shard in &pre.shards {
        let dest = stage.path().join(&shard.filename);
        let modified = shard.rewrite.is_some();
        let (sha256, payload, hardlinked) = if let Some(planned) = &shard.rewrite {
            (
                rewrite(shard, planned, &dest, plan)?,
                planned.payload_bytes,
                false,
            )
        } else {
            let linked = link_or_copy(
                &shard.path,
                &dest,
                plan.output_strategy == OutputStrategy::HardlinkUnmodifiedShards,
            )?;
            (
                hash_file(&dest)?,
                shard
                    .tensors
                    .values()
                    .map(|t| (t.data_offsets.1 - t.data_offsets.0) as u64)
                    .sum(),
                linked,
            )
        };
        total = total
            .checked_add(payload)
            .ok_or_else(|| format_error(&dest))?;
        let bytes = fs::metadata(&dest).map_err(|e| io_error(&dest, e))?.len();
        receipts.push(ShardReceipt {
            filename: shard.filename.clone(),
            modified,
            hardlinked,
            bytes,
            input_bytes: shard.bytes,
            byte_delta: delta(&dest, bytes, shard.bytes)?,
            sha256,
        });
    }
    for (name, source) in &pre.auxiliaries {
        let dest = stage.path().join(name);
        fs::copy(source, &dest).map_err(|e| io_error(&dest, e))?;
    }
    if let Some(config) = &pre.config {
        json_file(&stage.path().join("config.json"), config)?;
    }
    if let Some(index) = pre.index.as_mut() {
        index["metadata"]["total_size"] = total.into();
        json_file(&stage.path().join("model.safetensors.index.json"), index)?;
    }
    let receipt = PatchReceipt {
        sharded: pre.index.is_some(),
        total_tensor_bytes: total,
        total_input_tensor_bytes: pre.total_tensor_bytes,
        tensor_byte_delta: delta(&pre.output_dir, total, pre.total_tensor_bytes)?,
        tensors_modified: plan.edits.len(),
        shards: receipts,
    };
    if !pre.output_existed {
        fs::rename(stage.path(), &pre.output_dir).map_err(|e| io_error(&pre.output_dir, e))?;
    } else {
        publish_empty(stage.path(), &pre.output_dir, |a, b| fs::rename(a, b))?;
    }
    Ok(receipt)
}

fn publish_empty(
    stage: &Path,
    output: &Path,
    mut rename: impl FnMut(&Path, &Path) -> io::Result<()>,
) -> Result<(), SurgeonError> {
    let mut names: Vec<_> = fs::read_dir(stage)
        .map_err(|e| io_error(stage, e))?
        .map(|entry| entry.map(|e| e.file_name()))
        .collect::<Result<_, _>>()
        .map_err(|e| io_error(stage, e))?;
    names.sort_by_key(|name| {
        (
            name == "config.json" || name == "model.safetensors.index.json",
            name.clone(),
        )
    });
    let mut published = Vec::new();
    for name in names {
        let dest = output.join(&name);
        if let Err(e) = rename(&stage.join(&name), &dest) {
            let mut cleanup_error = None;
            for path in published {
                if let Err(error) = fs::remove_file(path) {
                    cleanup_error.get_or_insert(error);
                }
            }
            if let Some(error) = cleanup_error {
                return Err(io_error(output, error));
            }
            return Err(io_error(&dest, e));
        }
        published.push(dest);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached_header_matches_emitted_mixed_tensor_shard() {
        use crate::{checkpoint::preflight, RowFill, TensorEdit};
        use safetensors::{tensor::TensorView, Dtype, SafeTensors};
        let input = tempfile::tempdir().unwrap();
        let parent = tempfile::tempdir().unwrap();
        let dest = parent.path().join("output");
        let floats = 2f32.to_le_bytes();
        let half = half::f16::from_f32(3.).to_le_bytes();
        let nan = 0x7fc1u16.to_le_bytes();
        let tensors = [
            (
                "a",
                TensorView::new(Dtype::F32, vec![1, 1], &floats).unwrap(),
            ),
            (
                "b",
                TensorView::new(Dtype::U8, vec![3], &[7, 8, 9]).unwrap(),
            ),
            ("c", TensorView::new(Dtype::F16, vec![1, 1], &half).unwrap()),
            ("d", TensorView::new(Dtype::BF16, vec![1, 1], &nan).unwrap()),
        ];
        fs::write(
            input.path().join("model.safetensors"),
            safetensors::serialize(tensors, None).unwrap(),
        )
        .unwrap();
        let mut plan = CheckpointPatchPlan::default();
        assert!(preflight(input.path(), &dest, &plan).unwrap().shards[0]
            .rewrite
            .is_none());
        for name in ["a", "c"] {
            plan.insert_edit(
                name,
                TensorEdit::AddDense2D {
                    rows: 1,
                    cols: 1,
                    delta_row_major: vec![0.5],
                },
            )
            .unwrap();
        }
        plan.insert_edit(
            "d",
            TensorEdit::AppendRows2D {
                fill: RowFill::Values {
                    added_rows: 1,
                    data: vec![4.],
                },
            },
        )
        .unwrap();
        let pre = preflight(input.path(), &dest, &plan).unwrap();
        let planned = pre.shards[0].rewrite.as_ref().unwrap();
        let header = planned.header_bytes.clone();
        let payload_bytes = planned.payload_bytes;
        assert_eq!(header.len() % 8, 0);
        let receipt = apply(pre, &plan).unwrap();
        let bytes = fs::read(dest.join("model.safetensors")).unwrap();
        assert_eq!(
            u64::from_le_bytes(bytes[..8].try_into().unwrap()),
            header.len() as u64
        );
        assert_eq!(&bytes[8..8 + header.len()], header);
        assert_eq!(receipt.total_tensor_bytes, payload_bytes);
        assert_eq!(bytes.len() as u64, 8 + header.len() as u64 + payload_bytes);
        let parsed = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(parsed.tensor("a").unwrap().data(), 2.5f32.to_le_bytes());
        assert_eq!(parsed.tensor("b").unwrap().data(), [7, 8, 9]);
        assert_eq!(
            parsed.tensor("c").unwrap().data(),
            half::f16::from_f32(3.5).to_le_bytes()
        );
        let appended = [nan, half::bf16::from_f32(4.).to_le_bytes()].concat();
        assert_eq!(parsed.tensor("d").unwrap().data(), appended);
    }

    #[test]
    fn cross_device_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        let dest = dir.path().join("dest");
        fs::write(&source, b"payload").unwrap();
        assert!(!link_with(&source, &dest, true, |_, _| Err(io::Error::from(
            io::ErrorKind::CrossesDevices
        )))
        .unwrap());
        assert_eq!(fs::read(dest).unwrap(), b"payload");
        assert!(
            link_with(&source, &dir.path().join("denied"), true, |_, _| Err(
                io::Error::from(io::ErrorKind::PermissionDenied)
            ))
            .is_err()
        );
    }
    #[test]
    fn failed_publication_rolls_back() {
        let stage = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        fs::write(stage.path().join("a"), b"a").unwrap();
        fs::write(stage.path().join("b"), b"b").unwrap();
        let mut count = 0;
        assert!(publish_empty(stage.path(), out.path(), |a, b| {
            count += 1;
            if count == 2 {
                Err(io::Error::from(io::ErrorKind::PermissionDenied))
            } else {
                fs::rename(a, b)
            }
        })
        .is_err());
        assert_eq!(fs::read_dir(out.path()).unwrap().count(), 0);
    }
}
