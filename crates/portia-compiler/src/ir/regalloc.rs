//! Automatic Neural Register Allocator (`regalloc`).
//!
//! Walks an [`IntentExpr`] via stack-safe `collapse_frames` (`recursion`), verifies causal
//! upstream `<` downstream layer hops, propagates Rodney Brooks `high | low` priority-inhibition
//! edges, and deterministically allocates disjoint `[slot_lo .. slot_hi)` ranges per `(bank, layer)`.

use crate::error::V2CompilerError;
use crate::ir::{HardwareIntentKind, IntentExpr, IntentExprFrame, IntentRule};
use portia_core::hylo::CollapsibleExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Register-allocated rule ready for numerical lowering, carrying its automatically assigned
/// `[rel_slot_lo .. rel_slot_hi)` range and inherited priority-inhibitor rule names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllocatedIntentRule {
    pub rule: IntentRule,
    pub wire_bank_key: String,
    pub host_mlp_layer: usize,
    pub rel_slot_lo: usize,
    pub rel_slot_hi: usize,
    /// Names of higher-priority rules (`high | self`) at the same `host_mlp_layer` whose active
    /// cursor states are automatically enrolled in `self`'s negative `-8.0` clamp set!
    pub inhibited_by_rules: Vec<String>,
}

/// Complete register-allocation plan across an [`IntentExpr`] tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegisterAllocationPlan {
    pub allocated_rules: Vec<AllocatedIntentRule>,
    pub max_widened_slots_used: usize,
}

#[derive(Clone)]
struct SubtreeSummary {
    rules: Vec<(IntentRule, Vec<String>)>,
    rule_names_by_layer: BTreeMap<usize, Vec<String>>,
}

/// Validate causal layer hops and allocate disjoint SwiGLU slots up to `widened_slot_capacity`.
pub fn allocate_neural_registers(
    expr: &IntentExpr,
    widened_slot_capacity: usize,
) -> Result<RegisterAllocationPlan, V2CompilerError> {
    let summary: SubtreeSummary =
        expr.collapse_frames(|frame: IntentExprFrame<SubtreeSummary>| match frame {
            IntentExprFrame::Rule(r) => {
                let layer = r.kind.host_mlp_layer();
                let mut by_layer = BTreeMap::new();
                by_layer.insert(layer, vec![r.name.clone()]);
                SubtreeSummary {
                    rules: vec![(*r, Vec::new())],
                    rule_names_by_layer: by_layer,
                }
            }
            IntentExprFrame::Seq(mut lhs, rhs) | IntentExprFrame::Parallel(mut lhs, rhs) => {
                lhs.rules.extend(rhs.rules);
                for (l, names) in rhs.rule_names_by_layer {
                    lhs.rule_names_by_layer.entry(l).or_default().extend(names);
                }
                lhs
            }
            IntentExprFrame::Inhibit { mut high, mut low } => {
                // Every rule in `low` at layer `l` inherits inhibition from all rules in `high` at layer `l`:
                for (low_rule, inhib_list) in &mut low.rules {
                    let l = low_rule.kind.host_mlp_layer();
                    if let Some(high_names) = high.rule_names_by_layer.get(&l) {
                        for hn in high_names {
                            if !inhib_list.contains(hn) {
                                inhib_list.push(hn.clone());
                            }
                        }
                    }
                }
                high.rules.extend(low.rules);
                for (l, names) in low.rule_names_by_layer {
                    high.rule_names_by_layer.entry(l).or_default().extend(names);
                }
                high
            }
        });

    // Pass 1: Reserve `[0 .. static_floor[layer])` for `MountPolicy::PermanentStatic` rules first
    // so `permanent_static ∩ dynamic_banks == ∅` holds on every layer (`worker.py:335` invariant!):
    let mut static_floors: BTreeMap<usize, usize> = BTreeMap::new();
    for (rule, _) in &summary.rules {
        if matches!(rule.mount_policy, portia_core::MountPolicy::PermanentStatic) {
            let layer = rule.kind.host_mlp_layer();
            let span = rule.kind.required_slot_span();
            let floor = static_floors.entry(layer).or_insert(0);
            *floor += span;
            if *floor > widened_slot_capacity {
                return Err(V2CompilerError::SlotCapacityExhausted {
                    bank: "permanent_static".to_string(),
                    layer,
                    requested: *floor,
                    capacity: widened_slot_capacity,
                });
            }
        }
    }

    // Pass 2: Bump-allocate disjoint `[rel_slot_lo .. rel_slot_hi)` counters per `(wire_bank_key, host_mlp_layer)`,
    // starting dynamic banks strictly at `static_floors[layer]`:
    let mut static_cursors: BTreeMap<usize, usize> = BTreeMap::new();
    let mut dynamic_cursors: BTreeMap<(String, usize), usize> = BTreeMap::new();
    let mut allocated_rules = Vec::with_capacity(summary.rules.len());
    let mut max_used = 0usize;

    for (rule, inhibited_by_rules) in summary.rules {
        // Verify strict causal hop `upstream < downstream` on cross-layer intents:
        match &rule.kind {
            &HardwareIntentKind::SubmergedPointerRoute {
                upstream_mlp_layer,
                target_gqa_layer,
                ..
            }
            | &HardwareIntentKind::AsidSegmentColor {
                upstream_mlp_layer,
                target_gqa_layer,
                ..
            } => {
                if upstream_mlp_layer >= target_gqa_layer {
                    return Err(V2CompilerError::NonCausalLayerHop {
                        rule: rule.name.clone(),
                        upstream: upstream_mlp_layer,
                        downstream: target_gqa_layer,
                    });
                }
            }
            &HardwareIntentKind::DeltaNetRollbackScrub {
                upstream_mlp_layer,
                target_deltanet_layer,
                ..
            } => {
                if upstream_mlp_layer >= target_deltanet_layer {
                    return Err(V2CompilerError::NonCausalLayerHop {
                        rule: rule.name.clone(),
                        upstream: upstream_mlp_layer,
                        downstream: target_deltanet_layer,
                    });
                }
            }
            HardwareIntentKind::StanceRoute { .. }
            | HardwareIntentKind::SignedTernaryFork { .. } => {}
        }

        let bank_key = rule.mount_policy.wire_key().to_string();
        let host_mlp_layer = rule.kind.host_mlp_layer();
        let span = rule.kind.required_slot_span();
        let (rel_lo, rel_hi) =
            if matches!(rule.mount_policy, portia_core::MountPolicy::PermanentStatic) {
                let c = static_cursors.entry(host_mlp_layer).or_insert(0);
                let lo = *c;
                *c += span;
                (lo, *c)
            } else {
                let floor = static_floors.get(&host_mlp_layer).copied().unwrap_or(0);
                let c = dynamic_cursors
                    .entry((bank_key.clone(), host_mlp_layer))
                    .or_insert(floor);
                let lo = *c;
                *c += span;
                (lo, *c)
            };
        if rel_hi > widened_slot_capacity {
            return Err(V2CompilerError::SlotCapacityExhausted {
                bank: bank_key,
                layer: host_mlp_layer,
                requested: rel_hi,
                capacity: widened_slot_capacity,
            });
        }
        max_used = max_used.max(rel_hi);

        allocated_rules.push(AllocatedIntentRule {
            rule,
            wire_bank_key: bank_key,
            host_mlp_layer,
            rel_slot_lo: rel_lo,
            rel_slot_hi: rel_hi,
            inhibited_by_rules,
        });
    }

    Ok(RegisterAllocationPlan {
        allocated_rules,
        max_widened_slots_used: max_used,
    })
}
