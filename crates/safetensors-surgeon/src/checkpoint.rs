use crate::{mmap, output, tensor, CheckpointPatchPlan, PatchReceipt, SurgeonError};
use safetensors::tensor::TensorInfo;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

const INDEX: &str = "model.safetensors.index.json";
const AUXILIARIES: [&str; 5] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
];

pub(crate) struct Shard {
    pub filename: String,
    pub path: PathBuf,
    pub bytes: u64,
    pub data_start: u64,
    pub tensors: BTreeMap<String, TensorInfo>,
    pub metadata: Option<BTreeMap<String, String>>,
    pub rewrite: Option<PlannedRewrite>,
}

pub(crate) struct PlannedRewrite {
    pub header_bytes: Vec<u8>,
    pub payload_bytes: u64,
}

pub(crate) struct Preflight {
    pub output_dir: PathBuf,
    pub output_existed: bool,
    pub shards: Vec<Shard>,
    pub index: Option<Value>,
    pub config: Option<Value>,
    pub auxiliaries: Vec<(String, PathBuf)>,
    pub total_tensor_bytes: u64,
}

fn io_error(path: &Path, source: io::Error) -> SurgeonError {
    SurgeonError::Io {
        path: path.into(),
        source,
    }
}

fn invalid(path: &Path, message: &str) -> SurgeonError {
    io_error(path, io::Error::new(io::ErrorKind::InvalidInput, message))
}

fn format_error(path: &Path, message: impl Into<String>) -> SurgeonError {
    SurgeonError::SafetensorsFormat {
        path: path.into(),
        message: message.into(),
    }
}

fn json(path: &Path) -> Result<Value, SurgeonError> {
    let bytes = fs::read(path).map_err(|e| io_error(path, e))?;
    serde_json::from_slice(&bytes).map_err(|source| SurgeonError::Json {
        path: path.into(),
        source,
    })
}

fn exists(path: &Path) -> Result<bool, SurgeonError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(io_error(path, e)),
    }
}

fn regular_file(path: &Path) -> Result<fs::Metadata, SurgeonError> {
    let metadata = fs::metadata(path).map_err(|e| io_error(path, e))?;
    if !metadata.is_file() {
        return Err(invalid(path, "expected a regular file"));
    }
    Ok(metadata)
}

fn destination(input: &Path, output: &Path) -> Result<(PathBuf, bool), SurgeonError> {
    let existed = exists(output)?;
    let path = if existed {
        let metadata = fs::symlink_metadata(output).map_err(|e| io_error(output, e))?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(invalid(output, "output must be a directory, not a symlink"));
        }
        if fs::read_dir(output)
            .map_err(|e| io_error(output, e))?
            .next()
            .transpose()
            .map_err(|e| io_error(output, e))?
            .is_some()
        {
            return Err(invalid(output, "output directory must be empty"));
        }
        fs::canonicalize(output).map_err(|e| io_error(output, e))?
    } else {
        let name = output
            .file_name()
            .ok_or_else(|| invalid(output, "missing output directory name"))?;
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let parent = fs::canonicalize(parent).map_err(|e| io_error(parent, e))?;
        if !parent.is_dir() {
            return Err(invalid(&parent, "output parent must be a directory"));
        }
        parent.join(name)
    };
    if path.starts_with(input) || input.starts_with(&path) {
        return Err(invalid(
            output,
            "input and output directories must be distinct and not nested",
        ));
    }
    Ok((path, existed))
}

fn shard_name(name: &str) -> Result<(), SurgeonError> {
    let mut components = Path::new(name).components();
    let basename =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !basename
        || name.contains(['/', '\\', '\0', ':'])
        || name.contains("..")
        || name == INDEX
        || AUXILIARIES.contains(&name)
    {
        return Err(SurgeonError::UnsafeShardFilename(name.into()));
    }
    Ok(())
}

type Discovery = (Option<Value>, BTreeMap<String, String>, BTreeSet<String>);

fn discover(input: &Path) -> Result<Discovery, SurgeonError> {
    let path = input.join(INDEX);
    if !exists(&path)? {
        return Ok((
            None,
            BTreeMap::new(),
            BTreeSet::from(["model.safetensors".into()]),
        ));
    }
    let mut index = json(&path)?;
    let object = index
        .as_object_mut()
        .ok_or_else(|| format_error(&path, "index must be an object"))?;
    let weights = object
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| format_error(&path, "weight_map must be an object"))?;
    let mut weight_map = BTreeMap::new();
    let mut names = BTreeSet::new();
    for (name, value) in weights {
        let filename = value
            .as_str()
            .ok_or_else(|| format_error(&path, "weight_map values must be filenames"))?;
        shard_name(filename)?;
        weight_map.insert(name.clone(), filename.to_owned());
        names.insert(filename.to_owned());
    }
    let metadata = object
        .entry("metadata")
        .or_insert_with(|| Value::Object(Default::default()));
    if !metadata.is_object() {
        return Err(format_error(&path, "index metadata must be an object"));
    }
    Ok((Some(index), weight_map, names))
}

fn config(input: &Path, plan: &CheckpointPatchPlan) -> Result<Option<Value>, SurgeonError> {
    let Some(update) = &plan.config_intermediate_size else {
        return Ok(None);
    };
    let path = input.join("config.json");
    if !exists(&path)? {
        return Err(SurgeonError::ConfigMismatch("missing config.json".into()));
    }
    let mut value = json(&path)?;
    let error = || {
        SurgeonError::ConfigMismatch(
            "every present intermediate_size must match expected_old_size".into(),
        )
    };
    let root = value.as_object_mut().ok_or_else(error)?;
    let mut found = false;
    if let Some(size) = root.get_mut("intermediate_size") {
        if size.as_u64() != Some(update.expected_old_size as u64) {
            return Err(error());
        }
        *size = update.new_size.into();
        found = true;
    }
    if let Some(text) = root.get_mut("text_config") {
        let text = text.as_object_mut().ok_or_else(error)?;
        if let Some(size) = text.get_mut("intermediate_size") {
            if size.as_u64() != Some(update.expected_old_size as u64) {
                return Err(error());
            }
            *size = update.new_size.into();
            found = true;
        }
    }
    if !found {
        return Err(SurgeonError::ConfigMismatch(
            "missing intermediate_size".into(),
        ));
    }
    Ok(Some(value))
}

fn validate_plan(shards: &mut [Shard], plan: &CheckpointPatchPlan) -> Result<u64, SurgeonError> {
    let tensors: BTreeMap<_, _> = shards
        .iter()
        .flat_map(|s| s.tensors.iter().map(move |(n, t)| (n, (s, t))))
        .collect();
    for name in plan.edits.keys().chain(plan.expected_shapes.keys()) {
        if !tensors.contains_key(name) {
            return Err(SurgeonError::MissingTensor { name: name.clone() });
        }
    }
    for (name, expected) in &plan.expected_shapes {
        let shape = &tensors[name].1.shape;
        if shape.len() != 2 {
            return Err(SurgeonError::NotMatrix2D {
                name: name.clone(),
                actual: shape.clone(),
            });
        }
        if (shape[0], shape[1]) != *expected {
            return Err(SurgeonError::DimensionMismatch {
                name: name.clone(),
                expected: *expected,
                actual: (shape[0], shape[1]),
            });
        }
    }
    let mut input_total = 0u64;
    let mut output_total = 0u64;
    for shard in shards {
        let overflow = || format_error(&shard.path, "byte accounting overflow");
        let input_bytes = shard.bytes - shard.data_start;
        input_total = input_total.checked_add(input_bytes).ok_or_else(overflow)?;
        if !shard
            .tensors
            .keys()
            .any(|name| plan.edits.contains_key(name))
        {
            output_total = output_total.checked_add(input_bytes).ok_or_else(overflow)?;
            continue;
        }
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        if let Some(metadata) = &shard.metadata {
            header.insert(
                "__metadata__".into(),
                serde_json::to_value(metadata).expect("string map"),
            );
        }
        for (name, info) in &shard.tensors {
            let mut new = match plan.edits.get(name) {
                Some(edit) => tensor::validate(name, info, edit, &shard.path)?,
                None => info.clone(),
            };
            let end = offset
                .checked_add(new.data_offsets.1 - new.data_offsets.0)
                .ok_or_else(overflow)?;
            new.data_offsets = (offset, end);
            offset = end;
            header.insert(
                name.clone(),
                serde_json::to_value(new).expect("tensor info"),
            );
        }
        let mut header_bytes = serde_json::to_vec(&header).expect("header serialization");
        let header_len = header_bytes.len().checked_add(7).ok_or_else(overflow)? / 8 * 8;
        if header_len > 100_000_000 {
            return Err(format_error(
                &shard.path,
                "output header exceeds safetensors limit",
            ));
        }
        header_bytes.resize(header_len, b' ');
        let output_bytes = (offset as u64)
            .checked_add(header_len as u64)
            .and_then(|n| n.checked_add(8))
            .ok_or_else(overflow)?;
        i64::try_from(i128::from(output_bytes) - i128::from(shard.bytes))
            .map_err(|_| overflow())?;
        output_total = output_total
            .checked_add(offset as u64)
            .ok_or_else(overflow)?;
        shard.rewrite = Some(PlannedRewrite {
            header_bytes,
            payload_bytes: offset as u64,
        });
    }
    i64::try_from(i128::from(output_total) - i128::from(input_total))
        .map_err(|_| format_error(Path::new(INDEX), "total tensor byte delta exceeds i64"))?;
    Ok(input_total)
}

pub(crate) fn preflight(
    input: &Path,
    output: &Path,
    plan: &CheckpointPatchPlan,
) -> Result<Preflight, SurgeonError> {
    let input = fs::canonicalize(input).map_err(|e| io_error(input, e))?;
    if !input.is_dir() {
        return Err(invalid(&input, "input must be a directory"));
    }
    let (output_dir, output_existed) = destination(&input, output)?;
    let (index, weights, names) = discover(&input)?;
    let mut shards = Vec::new();
    let mut owners = BTreeMap::new();
    for filename in names {
        let path = input.join(&filename);
        let bytes = regular_file(&path)?.len();
        let (data_start, metadata) = mmap::read_metadata(&path)?;
        let tensors: BTreeMap<_, _> = metadata
            .tensors()
            .into_iter()
            .map(|(n, t)| (n, t.clone()))
            .collect();
        for name in tensors.keys() {
            if owners.insert(name.clone(), filename.clone()).is_some() {
                return Err(format_error(
                    &path,
                    format!("tensor {name:?} appears in multiple shards"),
                ));
            }
        }
        let metadata = metadata
            .metadata()
            .as_ref()
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
        shards.push(Shard {
            filename,
            path,
            bytes,
            data_start,
            tensors,
            metadata,
            rewrite: None,
        });
    }
    if index.is_some() && owners != weights {
        return Err(format_error(
            &input.join(INDEX),
            "weight_map must exactly match all tensors and their owning shards",
        ));
    }
    let total_tensor_bytes = validate_plan(&mut shards, plan)?;
    let config = config(&input, plan)?;
    let mut auxiliaries = Vec::new();
    for name in AUXILIARIES {
        if name == "config.json" && config.is_some() {
            continue;
        }
        let path = input.join(name);
        if exists(&path)? {
            regular_file(&path)?;
            auxiliaries.push((name.into(), path));
        }
    }
    Ok(Preflight {
        output_dir,
        output_existed,
        shards,
        index,
        config,
        auxiliaries,
        total_tensor_bytes,
    })
}

pub(crate) fn apply(
    input: &Path,
    output: &Path,
    plan: &CheckpointPatchPlan,
) -> Result<PatchReceipt, SurgeonError> {
    output::apply(preflight(input, output, plan)?, plan)
}
