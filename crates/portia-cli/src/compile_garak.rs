//! Pure-Rust CLI handler for `portia-cli compile-garak` (`crates/portia-cli/src/compile_garak.rs`).
//!
//! Lowers [`GarakCompileInputSpec`] through unified [`portia_compiler`] (`IntentExpr` `Parallel &`,
//! `StaticPermanentScope`, `StrictZeroCone` `AlphaEdit` preservation, and `faer` thin SVD) via
//! [`portia_garak::v2_concolic::compile_garak_v2_cartridge`], writing a scalar-checked
//! `permanent_static` [`WireCartridgeManifestJson`] candidate to `--out-json` and emitting
//! [`GarakCompileReceipt`] JSON to `stdout`.

#[cfg(test)]
mod tests;

use anyhow::{bail, Context, Result};
use portia_compiler::concolic::ConcolicDiodeLens;
use portia_compiler::wire::WireCartridgeManifestJson;
use portia_garak::v2_concolic::compile_garak_v2_cartridge;
use portia_garak::{
    resolve_garak_readout_payload, GarakCompileInputSpec, GarakCompileReceipt, GarakReadoutSpec,
    GarakResolvedReadouts,
};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

/// Options for [`compile_garak_command`] and [`compile_garak_v2_command`].
#[derive(Debug, Clone, Default)]
pub struct CompileGarakCliOverrides {
    pub slot_offset: Option<usize>,
    pub widened_slots: Option<usize>,
    pub max_bg_leakage: Option<f64>,
}

/// Resolve only `v_no` and `v_act` readout vectors from an input JSON containing either a direct
/// [`GarakReadoutSpec`] or a [`GarakCompileInputSpec`] (`--readout-only`).
pub fn resolve_garak_readout_command(input_json: &Path) -> Result<GarakResolvedReadouts> {
    let raw = std::fs::read_to_string(input_json).with_context(|| {
        format!(
            "failed to read Garak readout input JSON at {}",
            input_json.display()
        )
    })?;
    let spec: GarakReadoutSpec = match serde_json::from_str::<GarakReadoutSpec>(&raw) {
        Ok(direct) => direct,
        Err(_) => {
            let full: GarakCompileInputSpec = serde_json::from_str(&raw).with_context(|| {
                format!(
                    "failed to parse {} as GarakReadoutSpec or GarakCompileInputSpec",
                    input_json.display()
                )
            })?;
            full.readout
        }
    };
    resolve_garak_readout_payload(&spec)
        .map_err(|e| anyhow::anyhow!("Garak readout projection failed: {e}"))
}

fn load_and_apply_overrides(
    input_json: &Path,
    overrides: &CompileGarakCliOverrides,
) -> Result<GarakCompileInputSpec> {
    let raw = std::fs::read_to_string(input_json).with_context(|| {
        format!(
            "failed to read Garak compile input JSON at {}",
            input_json.display()
        )
    })?;
    let mut spec: GarakCompileInputSpec = serde_json::from_str(&raw).with_context(|| {
        format!(
            "failed to deserialize GarakCompileInputSpec from {}",
            input_json.display()
        )
    })?;

    if let Some(slot_off) = overrides.slot_offset {
        spec.slot_offset = Some(slot_off);
    }
    if let Some(widened) = overrides.widened_slots {
        if widened == 0 {
            bail!("--widened-slots must be > 0");
        }
        spec.widened_slots = Some(widened);
    }
    if let Some(leak_ceil) = overrides.max_bg_leakage {
        if !leak_ceil.is_finite() || leak_ceil < 0.0 {
            bail!("--max-bg-leakage must be finite and >= 0.0, got {leak_ceil}");
        }
        spec.max_bg_rss_leakage = leak_ceil;
        spec.strict_guard_check = true;
    }
    Ok(spec)
}

fn write_manifest_and_stamp_receipt(
    manifest: &WireCartridgeManifestJson,
    receipt: &mut GarakCompileReceipt,
    out_json: &Path,
    t0: Instant,
) -> Result<()> {
    if let Some(parent) = out_json.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create parent directory {} for Garak cartridge output",
                parent.display()
            )
        })?;
    }
    let manifest_json = manifest
        .to_json_pretty()
        .context("failed to serialize compiled Garak WireCartridgeManifestJson")?;
    std::fs::write(out_json, manifest_json).with_context(|| {
        format!(
            "failed to write compiled Garak WireCartridgeManifestJson to {}",
            out_json.display()
        )
    })?;
    receipt.out_json = Some(out_json.display().to_string());
    receipt.elapsed_ms = Some((t0.elapsed().as_secs_f64() * 1000.0 * 100.0).round() / 100.0);
    Ok(())
}

/// Compile [`GarakCompileInputSpec`] via unified `portia-compiler` (`IntentExpr` `Parallel &`,
/// `StrictZeroCone`, and `faer` thin SVD), returning `(manifest, receipt)`.
pub fn compile_garak_command(
    input_json: &Path,
    out_json: &Path,
    overrides: &CompileGarakCliOverrides,
) -> Result<(WireCartridgeManifestJson, GarakCompileReceipt)> {
    let (manifest, receipt, _lenses) = compile_garak_v2_command(input_json, out_json, overrides)?;
    Ok((manifest, receipt))
}

/// Compile [`GarakCompileInputSpec`] via unified `portia-compiler` (`IntentExpr` `Parallel &`,
/// `StrictZeroCone`, and `faer` thin SVD), returning `(manifest, receipt, concolic_lenses)`.
pub fn compile_garak_v2_command(
    input_json: &Path,
    out_json: &Path,
    overrides: &CompileGarakCliOverrides,
) -> Result<(
    WireCartridgeManifestJson,
    GarakCompileReceipt,
    BTreeMap<String, ConcolicDiodeLens>,
)> {
    let t0 = Instant::now();
    let spec = load_and_apply_overrides(input_json, overrides)?;
    let (manifest, mut receipt, lenses) = match compile_garak_v2_cartridge(&spec) {
        Ok(triple) => triple,
        Err(error) => {
            let rejection_path = out_json.with_extension("rejection.json");
            if let Some(parent) = rejection_path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
            {
                std::fs::create_dir_all(parent)?;
            }
            let rej_doc = serde_json::json!({
                "audit_status": "scalar_audit_rejected",
                "accepted": false,
                "compiler_engine": "v2",
                "model_id": spec.model_id,
                "candidate_index": spec.v2.as_ref().map(|v| v.candidate_index),
                "error": error,
            });
            std::fs::write(&rejection_path, serde_json::to_string_pretty(&rej_doc)?)?;
            bail!(
                "Garak compilation rejected: {error}; full diagnostics: {}",
                rejection_path.display()
            );
        }
    };
    write_manifest_and_stamp_receipt(&manifest, &mut receipt, out_json, t0)?;
    Ok((manifest, receipt, lenses))
}

/// Validate `--compiler-engine` CLI flag (only unified `"v2"` / `"pointer-isa"` is supported).
pub fn ensure_v2_compiler_engine_flag(raw: &str) -> Result<()> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "v2" | "pointer-isa" | "pointer_isa" | "faer" => Ok(()),
        other => bail!(
            "Unsupported --compiler-engine {other:?}; legacy v1 was removed—use unified v2 (`faer` + Clarabel Pointer-ISA)"
        ),
    }
}

/// Top-level `Commands::CompileGarak` handler.
pub fn run_compile_garak_subcommand<W: std::io::Write>(
    input_json: &Path,
    out_json: Option<&Path>,
    overrides: &CompileGarakCliOverrides,
    readout_only: bool,
    mut writer: W,
) -> Result<()> {
    if readout_only {
        let readouts = resolve_garak_readout_command(input_json)?;
        writeln!(writer, "{}", serde_json::to_string_pretty(&readouts)?)?;
    } else {
        let out_path = out_json.ok_or_else(|| {
            anyhow::anyhow!("--out-json is required unless --readout-only is passed")
        })?;
        let (_manifest, receipt, _lenses) =
            compile_garak_v2_command(input_json, out_path, overrides)?;
        writeln!(writer, "{}", serde_json::to_string_pretty(&receipt)?)?;
    }
    Ok(())
}
