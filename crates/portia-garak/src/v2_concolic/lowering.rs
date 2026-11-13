//! Declarative `IntentExpr` lowering (`Parallel &` across the 4 `GarakProbeFamily` obligations),
//! physical slot offset relocation, runtime `WireCartridgeManifestJson` conversion, and fail-closed
//! Cold Seam Audit (`guard_bg_max <= max_bg_rss_leakage`) via [`StaticPermanentScope`].

use crate::specs::{
    GarakCalibratedAngles, GarakCompileInputSpec, GarakCompileReceipt, GarakSlotDiagnosticCell,
    GarakSolverError, GARAK_SLOT_NAMES,
};
use crate::v2_concolic::splits::prepare_garak_v2_inputs;
use crate::v2_concolic::{CompiledGarakV2Output, GarakV2GeometryExtras};
use crate::GarakProbeFamily;
use portia_compiler::ir::scope_types::{ScopedIntentProgram, StaticPermanentScope};
use portia_compiler::ir::{HardwareIntentKind, IntentExpr, IntentRule};
use portia_compiler::lowering::{compile_intent_program, ModelGeometryTables};
use portia_compiler::numerics::sod::ChordalAngleRad;
use portia_compiler::numerics::{dot_f64, silu_f64};
use portia_compiler::wire::{
    WireBisimulationReceipt, WireCartridgeHeader, WireCartridgeManifestJson, WireSlotEntryJson,
};
use portia_core::pocket_dsl::RegimeExpr;
use portia_core::{BaseIntermediateSize, GrammarCartridgeSpec, MountPolicy, PhysicalSlotIndex};
use std::collections::BTreeMap;

/// Canonical 4-family `Garak` obligation order matching [`GARAK_SLOT_NAMES`] (`Slot0_GC..Slot3_PKG`).
pub const GARAK_V2_OBLIGATION_FAMILIES: [GarakProbeFamily; 4] = [
    GarakProbeFamily::SnowballGraphConnectivity,
    GarakProbeFamily::MisleadingFalseAssertion,
    GarakProbeFamily::SnowballSenators,
    GarakProbeFamily::PackageHallucination,
];

fn round_dec(val: f64, decimals: i32) -> f64 {
    let f = 10_f64.powi(decimals);
    let r = (val * f).round() / f;
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

/// Build the declarative `portia-compiler` [`IntentExpr`] composing all 4 [`GarakProbeFamily`]
/// static honesty obligations in parallel (`r0 & r1 & r2 & r3`) under [`MountPolicy::PermanentStatic`].
#[must_use]
pub fn build_garak_v2_intent_expr(
    target_layer_idx: usize,
    angles: GarakCalibratedAngles,
    targets: &[Vec<f64>; 4],
) -> IntentExpr {
    let thetas = [
        angles.theta_gc,
        angles.theta_fc,
        angles.theta_sen,
        angles.theta_pkg,
    ];
    let mut rules = GARAK_V2_OBLIGATION_FAMILIES
        .into_iter()
        .enumerate()
        .map(|(i, fam)| {
            IntentExpr::rule(IntentRule {
                name: fam.rule_name().to_string(),
                guard: RegimeExpr::ConstBool(true),
                mount_policy: MountPolicy::PermanentStatic,
                grammar: GrammarCartridgeSpec::Unconstrained,
                kind: HardwareIntentKind::StanceRoute {
                    layer_idx: target_layer_idx,
                    raw_target_dir: targets[i].clone(),
                    angle: ChordalAngleRad::new(thetas[i]),
                    expected_mode: None,
                },
            })
        });
    let first = rules.next().expect("4 Garak families");
    rules.fold(first, |acc, rhs| acc & rhs)
}

fn eval_coupled_pair_activations(entry: &WireSlotEntryJson, x_row: &[f64]) -> (f64, f64) {
    let g0: f64 = entry.w_gate[0]
        .iter()
        .zip(x_row)
        .map(|(&w, &x)| (w as f64) * x)
        .sum();
    let u_rad: f64 = entry.w_up[0]
        .iter()
        .zip(x_row)
        .map(|(&w, &x)| (w as f64) * x)
        .sum();
    let u_tan: f64 = entry
        .w_up
        .get(1)
        .unwrap_or(&entry.w_up[0])
        .iter()
        .zip(x_row)
        .map(|(&w, &x)| (w as f64) * x)
        .sum();
    let sg = silu_f64(g0);
    let act_rad = sg * u_rad;
    let act_tan = sg * u_tan;
    (act_tan, act_rad.hypot(act_tan))
}

pub fn lower_and_compile_garak_v2(
    spec: &GarakCompileInputSpec,
    extras: GarakV2GeometryExtras,
) -> Result<CompiledGarakV2Output, GarakSolverError> {
    if !spec.max_bg_rss_leakage.is_finite() || spec.max_bg_rss_leakage < 0.0 {
        return Err(GarakSolverError::InvalidSplit {
            name: "max_bg_rss_leakage".to_string(),
            reason: format!("must be finite and >= 0.0, got {}", spec.max_bg_rss_leakage),
        });
    }

    let prepared = prepare_garak_v2_inputs(spec)?;
    let d = prepared.hidden_dim;
    let base_m = spec.base_intermediate_size;
    let raw_offset = spec.slot_offset.unwrap_or(base_m);
    let (rel_lo_typed, abs_lo_typed) =
        PhysicalSlotIndex::resolve_flexible_boundary(raw_offset, BaseIntermediateSize(base_m));
    let rel_lo = rel_lo_typed.get();
    let abs_lo = abs_lo_typed.get();

    // 4 `StanceRoute` rules × 2 coupled `(radial, tangential)` slots per rule = 8 slots:
    let required_span = 8usize;
    let rel_hi = rel_lo + required_span;
    let abs_hi = abs_lo + required_span;
    let widened_slots = spec.widened_slots.unwrap_or_else(|| {
        if rel_lo == 0 {
            required_span
        } else {
            rel_hi.max(128)
        }
    });
    if rel_hi > widened_slots {
        return Err(GarakSolverError::SlotOffsetOutOfBounds {
            rel_lo,
            rel_hi,
            widened_slots,
        });
    }

    let scoped = ScopedIntentProgram::<StaticPermanentScope>::new_static(
        build_garak_v2_intent_expr(
            spec.target_layer_idx,
            spec.calibrated_angles,
            &prepared.target_dirs,
        ),
        spec.max_bg_rss_leakage,
    );
    let mut captures = BTreeMap::new();
    for (fam, cap) in GARAK_V2_OBLIGATION_FAMILIES
        .into_iter()
        .zip(prepared.captures_by_slot)
    {
        captures.insert(fam.rule_name().to_string(), cap);
    }

    let mut downstream_mlps = BTreeMap::new();
    if let Some(ds) = extras.downstream_mlp.or(prepared.spec_downstream_mlp) {
        downstream_mlps.insert(ds.layer_idx, ds);
    }
    let geom = ModelGeometryTables {
        model_id: spec.model_id.clone(),
        base_intermediate_size: base_m,
        widened_slot_capacity: widened_slots,
        w_u_top_basis: extras.w_u_top_basis,
        downstream_mlps,
        ..Default::default()
    };

    let program = compile_intent_program(&scoped.expr, &captures, &geom)
        .map_err(|e| GarakSolverError::ManifestIntegrity(format!("portia-compiler: {e}")))?;

    let layer_key = spec.target_layer_idx.to_string();
    let wire_entries = program
        .manifest
        .permanent_static
        .get(&layer_key)
        .ok_or_else(|| {
            GarakSolverError::ManifestIntegrity(format!(
                "missing permanent_static layer {layer_key} in CompiledProgramV2"
            ))
        })?;
    if wire_entries.len() != 4 {
        return Err(GarakSolverError::ManifestIntegrity(format!(
            "expected 4 compiled StanceRoute entries, got {}",
            wire_entries.len()
        )));
    }

    let relocated_entries: Vec<WireSlotEntryJson> = wire_entries
        .iter()
        .map(|e| WireSlotEntryJson {
            slot_lo: e.slot_lo + rel_lo,
            slot_hi: e.slot_hi + rel_lo,
            w_gate: e.w_gate.clone(),
            w_up: e.w_up.clone(),
            w_down: e.w_down.clone(),
        })
        .collect();

    let mut slot_diagnostics: BTreeMap<String, BTreeMap<String, GarakSlotDiagnosticCell>> =
        BTreeMap::new();
    let mut guard_bg_max = 0.0_f64;

    for (slot_idx, slot_name) in GARAK_SLOT_NAMES.iter().enumerate() {
        let entry = &relocated_entries[slot_idx];
        let mut cells = BTreeMap::new();
        for grp in &prepared.eval_groups {
            let n_rows = grp.mat.nrows().max(1);
            let mut sum_signed = 0.0_f64;
            let mut fire_cnt = 0usize;
            let mut max_mag = 0.0_f64;
            let mut worst_idx = 0usize;
            let mut worst_signed = 0.0_f64;

            for r in 0..grp.mat.nrows() {
                let (signed_tan, pair_mag) = eval_coupled_pair_activations(entry, grp.mat.row(r));
                sum_signed += signed_tan;
                if signed_tan > 0.25 {
                    fire_cnt += 1;
                }
                if pair_mag >= max_mag {
                    max_mag = pair_mag;
                    worst_idx = r;
                    worst_signed = signed_tan;
                }
            }
            let is_cold = grp.cold_for_slot[slot_idx];
            if is_cold {
                guard_bg_max = guard_bg_max.max(max_mag);
            }
            cells.insert(
                grp.key.clone(),
                GarakSlotDiagnosticCell {
                    fire_pct: round_dec((fire_cnt as f64 / n_rows as f64) * 100.0, 2),
                    mean: round_dec(sum_signed / (n_rows as f64), 4),
                    abs_max: round_dec(max_mag, 4),
                    worst_row_id: grp.row_ids.get(worst_idx).cloned(),
                    worst_signed_activation: round_dec(worst_signed, 4),
                    cold: is_cold,
                    rows: grp.mat.nrows(),
                },
            );
        }
        slot_diagnostics.insert((*slot_name).to_string(), cells);
    }

    if guard_bg_max > scoped.scope.max_bg_rss_leakage {
        return Err(GarakSolverError::GuardLeakageV2Exceeded {
            actual: guard_bg_max,
            ceiling: scoped.scope.max_bg_rss_leakage,
            diagnostics: slot_diagnostics,
        });
    }

    let bisim: Option<WireBisimulationReceipt> =
        program.manifest.bisimulation_receipt.as_ref().map(|b| {
            let _ = dot_f64(&prepared.target_dirs[0], &prepared.target_dirs[0]);
            b.clone()
        });

    let mut permanent_static = BTreeMap::new();
    permanent_static.insert(layer_key, relocated_entries);
    let manifest = WireCartridgeManifestJson {
        header: Some(WireCartridgeHeader {
            model_family: spec.model_id.clone(),
            checkpoint_revision: None,
            capture_sha256: None,
        }),
        model_id: Some(spec.model_id.clone()),
        base_intermediate_size: base_m,
        widened_slots,
        banks: BTreeMap::new(),
        permanent_static,
        mode_grammars: BTreeMap::new(),
        bisimulation_receipt: bisim,
    };
    manifest
        .verify_schema_integrity()
        .map_err(|e| GarakSolverError::ManifestIntegrity(e.to_string()))?;

    let receipt = GarakCompileReceipt {
        audit_status: "physical_audit_pending".into(),
        accepted: false,
        model_id: spec.model_id.clone(),
        target_layer: spec.target_layer_idx,
        hidden_dim: d,
        base_intermediate_size: base_m,
        widened_intermediate_size: base_m + widened_slots,
        widened_slots,
        slot_lo: abs_lo,
        slot_hi: abs_hi,
        scale_factor_sx: round_dec(prepared.scale_factor_sx, 4),
        calibrated_angles: spec.calibrated_angles,
        guard_bg_max: round_dec(guard_bg_max, 4),
        max_bg_rss_leakage: scoped.scope.max_bg_rss_leakage,
        guard_within_threshold: true,
        slot_diagnostics,
        out_json: None,
        elapsed_ms: None,
    };

    Ok(CompiledGarakV2Output {
        manifest,
        receipt,
        concolic_lenses: program.concolic_lenses.clone(),
        program,
    })
}
