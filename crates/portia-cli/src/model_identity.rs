//! Model and cartridge identity is deployment metadata, never task content.
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum ModelIdentityError {
    #[error("MissingCartridge: no SQL cartridge for {model}; pass --cartridge <path> or --cartridge none")]
    MissingCartridge { model: String },
    #[error("UnknownModelFamily: {model}; provide --model-family <org/repo> or --allow-cartridge-model-mismatch")]
    UnknownModelFamily { model: String },
    #[error("CartridgeModelMismatch: requested {requested:?}, cartridge {cartridge:?} at {path}; use --allow-cartridge-model-mismatch only for an intentional experiment")]
    CartridgeModelMismatch {
        requested: Option<String>,
        cartridge: Option<String>,
        path: String,
    },
    #[error("InvalidModelFamily: expected canonical org/repo identity, got {0:?}")]
    InvalidModelFamily(String),
    #[error("ModelFamilyConflict: declared identity {declared:?} contradicts known model identity {discovered:?}; --model-family may only identify an otherwise unknown model")]
    ModelFamilyConflict {
        declared: String,
        discovered: String,
    },
}

#[derive(Debug, Clone, Default)]
pub struct WorkerLaunchOptions {
    pub model_family: Option<String>,
    pub allow_cartridge_model_mismatch: bool,
    pub trace_jsonl: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct RuntimeProvenance {
    pub requested_model: String,
    pub resolved_model_family: Option<String>,
    pub cartridge_model_family: Option<String>,
    pub cartridge_sha256: Option<String>,
    pub cartridge_path: Option<String>,
    pub model_mismatch_override: bool,
    pub cartridge_mounted: bool,
    pub actual_model_loaded: bool,
    pub checkpoint_revision: Option<String>,
    pub resolved_model_path: Option<String>,
    pub device: Option<String>,
    pub dtype: Option<String>,
}

pub fn normalize_model_identifier(raw: &str) -> String {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let short = lower
        .strip_prefix("qwen3.5-")
        .or_else(|| lower.strip_prefix("qwen-"))
        .unwrap_or(&lower);
    let (size, base) = short
        .strip_suffix("-base")
        .map_or((short, false), |s| (s, true));
    let canonical_size = match size {
        "0.8b" => Some("0.8B"),
        "2b" => Some("2B"),
        "4b" => Some("4B"),
        "9b" => Some("9B"),
        "27b" => Some("27B"),
        _ => None,
    };
    canonical_size.map_or_else(
        || trimmed.to_string(),
        |size| format!("Qwen/Qwen3.5-{size}{}", if base { "-Base" } else { "" }),
    )
}

fn canonical_repo_id(s: &str) -> bool {
    let parts: Vec<_> = s.split('/').collect();
    parts.len() == 2
        && parts.iter().all(|p| {
            !p.is_empty()
                && *p != "."
                && *p != ".."
                && p.chars()
                    .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        })
}

fn extract_hf_snapshot_repo_id(raw_path: &str) -> Option<String> {
    let components: Vec<_> = Path::new(raw_path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect();
    for window in components.windows(3) {
        if window[1] == "snapshots" {
            if let Some(repo) = window[0].strip_prefix("models--") {
                if let Some((org, name)) = repo.split_once("--") {
                    let family = format!("{org}/{name}");
                    if canonical_repo_id(&family) {
                        return Some(family);
                    }
                }
            }
        }
    }
    for comp in &components {
        if let Some(repo) = comp.strip_prefix("models--") {
            if let Some((org, name)) = repo.split_once("--") {
                let family = format!("{org}/{name}");
                if canonical_repo_id(&family) {
                    return Some(family);
                }
            }
        }
    }
    None
}

/// Lightweight cartridge identity probe that skips deserializing the `411k`-float `banks` and
/// `permanent_static` weight matrices during pre-spawn provenance inspection.
#[derive(Deserialize)]
struct CartridgeIdentityProbe {
    #[serde(default)]
    header: Option<portia_compiler::wire::WireCartridgeHeader>,
    #[serde(default)]
    model_id: Option<String>,
    #[allow(dead_code)]
    widened_slots: usize,
    #[allow(dead_code)]
    banks: serde::de::IgnoredAny,
}

impl CartridgeIdentityProbe {
    fn model_family(&self) -> Result<Option<&str>, portia_compiler::error::V2CompilerError> {
        let identities = [
            self.header.as_ref().map(|h| h.model_family.as_str()),
            self.model_id.as_deref(),
        ];
        let mut family = None;
        for identity in identities.into_iter().flatten() {
            if identity.trim().is_empty() || family.is_some_and(|previous| previous != identity) {
                return Err(portia_compiler::error::V2CompilerError::WireIntegrity(
                    "conflicting or empty cartridge model identities (header.model_family, model_id)".into(),
                ));
            }
            family = Some(identity);
        }
        Ok(family)
    }
}

pub fn resolve_model_family(model: &str, explicit: Option<&str>) -> Result<Option<String>> {
    let declared = if let Some(family) = explicit {
        let family = normalize_model_identifier(family);
        if !canonical_repo_id(&family) {
            return Err(ModelIdentityError::InvalidModelFamily(family).into());
        }
        Some(family)
    } else {
        None
    };
    let model = normalize_model_identifier(model);
    let path = Path::new(&model);
    let mut identities = Vec::new();
    if !path.exists() && canonical_repo_id(&model) {
        identities.push(model.clone());
    }
    if let Some(snapshot_family) = extract_hf_snapshot_repo_id(&model) {
        identities.push(snapshot_family);
    }
    for (file, key) in [
        ("portia_model_provenance.json", "model_family"),
        ("config.json", "_name_or_path"),
    ] {
        let source = path.join(file);
        if source.is_file() {
            let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&source)?)
                .with_context(|| format!("invalid model provenance {}", source.display()))?;
            if let Some(val_str) = raw.get(key).and_then(serde_json::Value::as_str) {
                let normalized_val = normalize_model_identifier(val_str);
                if canonical_repo_id(&normalized_val) {
                    identities.push(normalized_val);
                } else if let Some(snapshot_family) = extract_hf_snapshot_repo_id(val_str) {
                    identities.push(snapshot_family);
                }
            }
        }
    }
    let mut inferred: Option<String> = None;
    for identity in identities {
        if let Some(previous) = &inferred {
            if previous != &identity {
                return Err(ModelIdentityError::ModelFamilyConflict {
                    declared: previous.clone(),
                    discovered: identity,
                }
                .into());
            }
        } else {
            inferred = Some(identity);
        }
    }
    if let (Some(declared), Some(discovered)) = (&declared, &inferred) {
        if declared != discovered {
            return Err(ModelIdentityError::ModelFamilyConflict {
                declared: declared.clone(),
                discovered: discovered.clone(),
            }
            .into());
        }
    }
    Ok(inferred.or(declared))
}

pub fn default_cartridge_filename(family: Option<&str>) -> Option<&'static str> {
    match family {
        Some("Qwen/Qwen3.5-0.8B") => Some("portia_v3_unified_cartridge.json"),
        Some("Qwen/Qwen3.5-0.8B-Base") => Some("portia_v3_unified_cartridge_08b_base.json"),
        Some("Qwen/Qwen3.5-2B-Base") => Some("portia_v3_unified_cartridge_2b_base.json"),
        Some("Qwen/Qwen3.5-4B-Base") => Some("portia_v3_unified_cartridge_4b_base.json"),
        Some("Qwen/Qwen3.5-9B-Base") => Some("portia_v3_unified_cartridge_9b_base.json"),
        _ => None,
    }
}

pub fn inspect_runtime_provenance(
    model: &str,
    cartridge: &Path,
    options: &WorkerLaunchOptions,
) -> Result<RuntimeProvenance> {
    let family = resolve_model_family(model, options.model_family.as_deref())?;
    let mut result = RuntimeProvenance {
        requested_model: model.to_string(),
        resolved_model_family: family.clone(),
        model_mismatch_override: options.allow_cartridge_model_mismatch,
        ..Default::default()
    };
    if crate::worker::is_unpatched_base_cartridge_sentinel(cartridge) {
        return Ok(result);
    }
    let bytes = std::fs::read(cartridge)
        .with_context(|| format!("reading cartridge {}", cartridge.display()))?;
    let probe: CartridgeIdentityProbe = serde_json::from_slice(&bytes).map_err(|e| {
        portia_compiler::error::V2CompilerError::WireIntegrity(format!(
            "Failed to deserialize cartridge identity header at {}: {e}",
            cartridge.display()
        ))
    })?;
    let artifact_family = probe.model_family()?.map(str::to_string);
    if !options.allow_cartridge_model_mismatch {
        if family.is_none() {
            return Err(ModelIdentityError::UnknownModelFamily {
                model: model.into(),
            }
            .into());
        }
        if artifact_family != family {
            return Err(ModelIdentityError::CartridgeModelMismatch {
                requested: family,
                cartridge: artifact_family,
                path: cartridge.display().to_string(),
            }
            .into());
        }
    }
    result.cartridge_path = Some(cartridge.display().to_string());
    result.cartridge_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
    result.cartridge_model_family = artifact_family;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aliases_and_local_provenance_are_explicit() {
        assert_eq!(normalize_model_identifier("0.8b"), "Qwen/Qwen3.5-0.8B");
        assert_eq!(
            normalize_model_identifier("qwen-4b-base"),
            "Qwen/Qwen3.5-4B-Base"
        );
        assert_eq!(
            resolve_model_family("/cache/models--Qwen--Qwen3.5-0.8B-Base/snapshots/abc", None)
                .unwrap()
                .as_deref(),
            Some("Qwen/Qwen3.5-0.8B-Base")
        );
        let temp = tempfile::tempdir().unwrap();
        assert_eq!(
            resolve_model_family(temp.path().to_str().unwrap(), None).unwrap(),
            None
        );
        assert_eq!(
            resolve_model_family(temp.path().to_str().unwrap(), Some("0.8b"))
                .unwrap()
                .as_deref(),
            Some("Qwen/Qwen3.5-0.8B")
        );
        std::fs::write(
            temp.path().join("config.json"),
            r#"{"_name_or_path":"Qwen/Qwen3.5-0.8B-Base"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_model_family(temp.path().to_str().unwrap(), None)
                .unwrap()
                .as_deref(),
            Some("Qwen/Qwen3.5-0.8B-Base")
        );
        std::fs::write(
            temp.path().join("config.json"),
            r#"{"_name_or_path":"/home/alice/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/0123abcdef"}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_model_family(temp.path().to_str().unwrap(), None)
                .unwrap()
                .as_deref(),
            Some("Qwen/Qwen3.5-0.8B-Base")
        );
    }

    #[test]
    fn explicit_model_family_cannot_relabel_a_known_checkpoint() {
        let error = resolve_model_family("0.8b-base", Some("0.8b")).unwrap_err();
        assert!(matches!(
            error.downcast_ref::<ModelIdentityError>(),
            Some(ModelIdentityError::ModelFamilyConflict { .. })
        ));
        assert_eq!(
            resolve_model_family("0.8b-base", Some("0.8b-base"))
                .unwrap()
                .as_deref(),
            Some("Qwen/Qwen3.5-0.8B-Base")
        );
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join("config.json"),
            r#"{"_name_or_path":"Qwen/Qwen3.5-0.8B-Base"}"#,
        )
        .unwrap();
        assert!(resolve_model_family(temp.path().to_str().unwrap(), Some("0.8b")).is_err());
        std::fs::write(
            temp.path().join("portia_model_provenance.json"),
            r#"{"model_family":"Qwen/Qwen3.5-0.8B"}"#,
        )
        .unwrap();
        assert!(resolve_model_family(temp.path().to_str().unwrap(), None).is_err());
        // Cartridge override never grants permission to falsify checkpoint provenance.
        let options = WorkerLaunchOptions {
            model_family: Some("0.8b".into()),
            allow_cartridge_model_mismatch: true,
            ..Default::default()
        };
        assert!(inspect_runtime_provenance("0.8b-base", Path::new("none"), &options).is_err());
    }
}
