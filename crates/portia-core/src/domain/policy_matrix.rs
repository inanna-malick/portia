//! Multi-clause coupled policy predicate matrices (`CompoundPolicyMatrix`) for nursery scenarios.

use super::governance::RowPolicyDecision;
use serde::{Deserialize, Serialize};

/// Operator type for a single clause in a [`CompoundPolicyMatrix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyClauseKind {
    /// Column value must be `>= threshold` (`col >= threshold`).
    AtLeastThreshold { threshold: i64 },
    /// Column value must be `<= ceiling` (`col <= ceiling`).
    AtMostCeiling { ceiling: i64 },
    /// Boolean flag column must equal `expected_bit` (`0` or `1`).
    BooleanFlagEquals { expected_bit: i64 },
    /// Split-column sum or tier check (`col_val % modulus == remainder`).
    TierModulusCheck { modulus: i64, remainder: i64 },
}

/// A single guard clause inside [`CompoundPolicyMatrix`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyGuardClause {
    pub clause_index: usize,
    pub column_name: String,
    pub kind: PolicyClauseKind,
    /// Whether violating this clause forces `Deny` (`true`) or `Ask` (`false`).
    pub hard_deny_on_violation: bool,
}

impl PolicyGuardClause {
    /// Evaluate this single guard clause on `value`.
    pub const fn evaluate(&self, value: i64) -> bool {
        match self.kind {
            PolicyClauseKind::AtLeastThreshold { threshold } => value >= threshold,
            PolicyClauseKind::AtMostCeiling { ceiling } => value <= ceiling,
            PolicyClauseKind::BooleanFlagEquals { expected_bit } => (value & 1) == expected_bit,
            PolicyClauseKind::TierModulusCheck { modulus, remainder } => {
                let m = if modulus <= 0 { 2 } else { modulus };
                value.rem_euclid(m) == remainder.rem_euclid(m)
            }
        }
    }

    /// Return a canonical satisfying integer value for this clause.
    pub const fn satisfying_value(&self) -> i64 {
        match self.kind {
            PolicyClauseKind::AtLeastThreshold { threshold } => threshold + 5,
            PolicyClauseKind::AtMostCeiling { ceiling } => ceiling - 5,
            PolicyClauseKind::BooleanFlagEquals { expected_bit } => expected_bit & 1,
            PolicyClauseKind::TierModulusCheck {
                modulus: _,
                remainder,
            } => remainder,
        }
    }

    /// Return a canonical violating integer value for this clause.
    pub const fn violating_value(&self) -> i64 {
        match self.kind {
            PolicyClauseKind::AtLeastThreshold { threshold } => threshold - 10,
            PolicyClauseKind::AtMostCeiling { ceiling } => ceiling + 10,
            PolicyClauseKind::BooleanFlagEquals { expected_bit } => (expected_bit & 1) ^ 1,
            PolicyClauseKind::TierModulusCheck { modulus, remainder } => {
                let m = if modulus <= 1 { 2 } else { modulus };
                (remainder + 1).rem_euclid(m)
            }
        }
    }
}

/// Multi-clause coupled policy predicate matrix (`PolicyPredicateArity` `1..=16`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompoundPolicyMatrix {
    pub clauses: Vec<PolicyGuardClause>,
}

impl CompoundPolicyMatrix {
    /// Deterministic truth-table evaluation of a row's policy guard columns:
    /// - If any `hard_deny_on_violation` clause fails -> [`RowPolicyDecision::Deny`].
    /// - Else if any soft steward clause fails -> [`RowPolicyDecision::Ask`].
    /// - Else (all `K` clauses satisfied) -> [`RowPolicyDecision::Allow`].
    pub fn evaluate(&self, row_values: &[i64]) -> RowPolicyDecision {
        let mut any_soft_ask = false;
        for (idx, clause) in self.clauses.iter().enumerate() {
            let val = row_values.get(idx).copied().unwrap_or(0);
            if !clause.evaluate(val) {
                if clause.hard_deny_on_violation {
                    return RowPolicyDecision::Deny;
                } else {
                    any_soft_ask = true;
                }
            }
        }
        if any_soft_ask {
            RowPolicyDecision::Ask
        } else {
            RowPolicyDecision::Allow
        }
    }

    /// Construct a deterministic row vector of length `clauses.len()` guaranteed to evaluate to `target`.
    pub fn synthesize_witness_row(&self, target: RowPolicyDecision) -> Option<Vec<i64>> {
        let mut row: Vec<i64> = self
            .clauses
            .iter()
            .map(PolicyGuardClause::satisfying_value)
            .collect();
        match target {
            RowPolicyDecision::Allow => Some(row),
            RowPolicyDecision::Ask => {
                let soft_idx = self
                    .clauses
                    .iter()
                    .position(|c| !c.hard_deny_on_violation)?;
                row[soft_idx] = self.clauses[soft_idx].violating_value();
                Some(row)
            }
            RowPolicyDecision::Deny => {
                let hard_idx = self.clauses.iter().position(|c| c.hard_deny_on_violation)?;
                row[hard_idx] = self.clauses[hard_idx].violating_value();
                Some(row)
            }
        }
    }
}

/// Generate a coupled `clause_arity` (`1..=16`) policy matrix with both soft-ask and hard-deny guards.
pub fn generate_compound_policy_matrix(clause_arity: usize, seed: u64) -> CompoundPolicyMatrix {
    let k = clause_arity.clamp(1, 16);
    let mut clauses = Vec::with_capacity(k);
    for i in 0..k {
        let mix = seed.wrapping_add((i as u64).wrapping_mul(0x9E3779B97F4A7C15));
        let kind = match i % 4 {
            0 => PolicyClauseKind::BooleanFlagEquals {
                expected_bit: (mix & 1) as i64,
            },
            1 => PolicyClauseKind::AtLeastThreshold {
                threshold: 100 + ((mix % 40) as i64) * 10,
            },
            2 => PolicyClauseKind::AtMostCeiling {
                ceiling: 800 - ((mix % 30) as i64) * 10,
            },
            _ => PolicyClauseKind::TierModulusCheck {
                modulus: 3,
                remainder: (mix % 3) as i64,
            },
        };
        let hard_deny = if i == 0 {
            false
        } else if i == 1 {
            true
        } else {
            (mix & 2) != 0
        };
        clauses.push(PolicyGuardClause {
            clause_index: i,
            column_name: format!("policy_guard_c{i}"),
            kind,
            hard_deny_on_violation: hard_deny,
        });
    }
    CompoundPolicyMatrix { clauses }
}
