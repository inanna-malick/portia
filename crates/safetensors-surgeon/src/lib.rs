//! `safetensors-surgeon` — Bounded-memory streaming `.safetensors` shard editor
//! for 2D row/column widening and additive weight patches without PyTorch.
//!
//! Input files and symlink targets must remain immutable throughout an operation:
//! external writes or truncation invalidate the safety of file-backed mappings.
//! Payload processing uses 8 MiB mappings and 1 MiB buffered writes, excluding
//! caller-owned edit vectors and checkpoint metadata from the working-memory bound.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

mod checkpoint;
mod mmap;
mod output;
mod plan;
mod tensor;

use safetensors::Dtype;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors returned during plan validation or streaming `.safetensors` surgery.
#[derive(Debug, Error)]
pub enum SurgeonError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("safetensors format error in {path}: {message}")]
    SafetensorsFormat { path: PathBuf, message: String },

    #[error("JSON error in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },

    #[error("unsafe shard filename in weight_map: {0:?}")]
    UnsafeShardFilename(String),

    #[error("tensor {name:?} not found in checkpoint")]
    MissingTensor { name: String },

    #[error("duplicate edit targeting tensor {name:?}")]
    DuplicateTensorTarget { name: String },

    #[error("unsupported dtype {dtype:?} for float edit on tensor {name:?} (expected BF16, F16, or F32)")]
    UnsupportedDtype { name: String, dtype: Dtype },

    #[error("expected 2D tensor for {name:?}, found shape {actual:?}")]
    NotMatrix2D { name: String, actual: Vec<usize> },

    #[error("dimension mismatch on tensor {name:?}: expected {expected:?}, got {actual:?}")]
    DimensionMismatch {
        name: String,
        expected: (usize, usize),
        actual: (usize, usize),
    },

    #[error("invalid edit buffer length for {name:?}: expected {expected} elements, got {actual}")]
    BufferLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("non-finite value (NaN or Inf) in edit input or patched output on tensor {name:?} at ({row}, {col})")]
    NonFiniteResult {
        name: String,
        row: usize,
        col: usize,
    },

    #[error("config.json validation error: {0}")]
    ConfigMismatch(String),
}

/// Payload source when appending `K` rows (`[R, C] -> [R + K, C]`).
#[derive(Debug, Clone, PartialEq)]
pub enum RowFill {
    /// Append `added_rows` zero-initialized rows (`added_rows > 0`).
    Zeros { added_rows: usize },
    /// Append `added_rows` explicit row-major `f32` values (`data.len() == added_rows * cols`).
    Values { added_rows: usize, data: Vec<f32> },
}

/// Payload source when appending `K` columns to every row (`[R, C] -> [R, C + K]`).
#[derive(Debug, Clone, PartialEq)]
pub enum ColFill {
    /// Append `added_cols` zero-initialized columns to each row (`added_cols > 0`).
    Zeros { added_cols: usize },
    /// Append `added_cols` explicit row-major `[rows, added_cols]` `f32` values (`data.len() == rows * added_cols`).
    Values { added_cols: usize, data: Vec<f32> },
}

/// A single 2D matrix transformation applied during streaming shard rewrite.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorEdit {
    /// Shape-preserving additive delta: `W'[r, c] = W[r, c] + delta[r * cols + c]`.
    AddDense2D {
        rows: usize,
        cols: usize,
        delta_row_major: Vec<f32>,
    },
    /// Row-widening edit: `[R, C] -> [R + K, C]` (preserves `[0..R, 0..C]` bytes verbatim).
    AppendRows2D { fill: RowFill },
    /// Column-widening edit: `[R, C] -> [R, C + K]` (preserves each row's `[0..C]` bytes verbatim).
    AppendCols2D { fill: ColFill },
}

/// Strategy for handling multi-shard files that contain zero edited tensors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputStrategy {
    /// Copy unmodified shards byte-for-byte.
    #[default]
    CopyUnmodifiedShards,
    /// Attempt `std::fs::hard_link` for unmodified shards, falling back to byte copy across filesystems.
    HardlinkUnmodifiedShards,
}

/// Optional `config.json` update applied alongside tensor edits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfigIntermediateSizeUpdate {
    pub expected_old_size: usize,
    pub new_size: usize,
}

/// Complete declarative patch plan over a single-file or sharded `.safetensors` directory.
#[derive(Debug, Clone, Default)]
pub struct CheckpointPatchPlan {
    pub edits: BTreeMap<String, TensorEdit>,
    /// Optional exact input matrix shapes, checked before any output is written.
    pub expected_shapes: BTreeMap<String, (usize, usize)>,
    pub output_strategy: OutputStrategy,
    pub config_intermediate_size: Option<ConfigIntermediateSizeUpdate>,
}

impl CheckpointPatchPlan {
    /// Insert an edit without silently replacing an existing target.
    ///
    /// Direct insertion into the public `edits` map has ordinary map replacement
    /// semantics; use this method when duplicate targets should be rejected.
    ///
    /// ```
    /// use safetensors_surgeon::{CheckpointPatchPlan, RowFill, TensorEdit};
    ///
    /// let mut plan = CheckpointPatchPlan::default();
    /// plan.insert_edit("weight", TensorEdit::AppendRows2D {
    ///     fill: RowFill::Zeros { added_rows: 2 },
    /// })?;
    /// assert_eq!(plan.edits.len(), 1);
    /// # Ok::<(), safetensors_surgeon::SurgeonError>(())
    /// ```
    pub fn insert_edit(
        &mut self,
        name: impl Into<String>,
        edit: TensorEdit,
    ) -> Result<(), SurgeonError> {
        match self.edits.entry(name.into()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(edit);
                Ok(())
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                Err(SurgeonError::DuplicateTensorTarget {
                    name: entry.key().clone(),
                })
            }
        }
    }
}

/// Explicit per-layer MLP slot bundle (`gate_proj`, `up_proj`, `down_proj`) for [`MlpWidenPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct LayerMlpAppendSpec {
    pub layer_index: usize,
    /// Shape `[added_slots, hidden_size]` row-major.
    pub gate_rows: Vec<f32>,
    /// Shape `[added_slots, hidden_size]` row-major.
    pub up_rows: Vec<f32>,
    /// Shape `[hidden_size, added_slots]` row-major.
    pub down_cols: Vec<f32>,
}

/// High-level builder for uniform `M -> M + K` HuggingFace SwiGLU MLP widening.
#[derive(Debug, Clone)]
pub struct MlpWidenPlan {
    /// Layer prefix up to `{layer_idx}` (e.g. `"model.layers"` or `"model.language_model.layers"`).
    pub layer_prefix: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub base_intermediate_size: usize,
    pub added_slots: usize,
    pub target_layers: Vec<LayerMlpAppendSpec>,
    pub uniform_across_all_layers: bool,
    pub output_strategy: OutputStrategy,
}

impl MlpWidenPlan {
    /// Lower this MLP widening specification into a validated [`CheckpointPatchPlan`].
    pub fn into_patch_plan(self) -> Result<CheckpointPatchPlan, SurgeonError> {
        plan::lower(self)
    }
}

/// Per-shard provenance emitted by [`apply_checkpoint_patch`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardReceipt {
    pub filename: String,
    pub modified: bool,
    pub hardlinked: bool,
    /// Entire input shard size, including its header.
    pub input_bytes: u64,
    /// Entire output shard size, including its header.
    pub bytes: u64,
    /// Signed change in the entire shard size (`bytes - input_bytes`).
    pub byte_delta: i64,
    pub sha256: String,
}

/// Deterministic cryptographic & structural receipt emitted after patching a checkpoint directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatchReceipt {
    pub sharded: bool,
    /// Sum of input tensor payload lengths, excluding headers.
    pub total_input_tensor_bytes: u64,
    pub total_tensor_bytes: u64,
    /// Signed change in total tensor payload bytes, excluding headers.
    pub tensor_byte_delta: i64,
    pub tensors_modified: usize,
    pub shards: Vec<ShardReceipt>,
}

/// Validate `plan` against `input_dir` and stream the patched checkpoint into `output_dir`.
///
/// Works on both single-file (`model.safetensors`) and multi-shard (`model.safetensors.index.json`)
/// HuggingFace directories with an 8 MiB input window and a 1 MiB output buffer.
/// Caller-owned edit vectors and checkpoint metadata are outside this working-memory bound.
/// The output directory must be absent or empty. Inputs, including symlink targets,
/// must remain immutable throughout the operation.
pub fn apply_checkpoint_patch(
    input_dir: &Path,
    output_dir: &Path,
    plan: &CheckpointPatchPlan,
) -> Result<PatchReceipt, SurgeonError> {
    checkpoint::apply(input_dir, output_dir, plan)
}
