//! Single-bug AST mutant paramorphism (`single_bug_mutants`) and structured mutant diagnosis.

use std::collections::BTreeMap;

use portia_core::hylo::CollapsibleExt;
use serde::{Deserialize, Serialize};

use super::predicate::{SqlPredicateAst, SqlPredicateFrame};
use super::scalar_ops::SqlComparisonOp;

/// Basis on which a candidate SQL predicate matched a single-bug AST mutant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutantMatchBasis {
    /// Candidate predicate's normalized AST / canonical SQL (`render_sql()`) matched the mutant identically.
    ExactNormalizedAst,
    /// Candidate predicate differed structurally from the mutant, but produced the exact same
    /// boolean truth table across all `sample_row_count` supplied sample rows.
    SampleRowEquivalence { sample_row_count: usize },
}

/// Structured diagnosis returned by [`SqlPredicateAst::diagnose_single_bug_mutant`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutantMatchDiagnosis {
    pub mutant_name: String,
    pub mutant_sql: String,
    pub match_basis: MutantMatchBasis,
}

impl SqlPredicateAst {
    /// Single-pass paramorphism via [`CollapsibleExt::collapse_frames`] generating named
    /// single-bug AST mutants (`dropped_left_conjunct`, `dropped_right_conjunct`,
    /// `exclusive_boundary`, `negated_null_check`, `and_to_or`).
    pub fn single_bug_mutants(&self) -> Vec<(String, SqlPredicateAst)> {
        let (_reconstructed, mutants) = self.collapse_frames(
            |frame: SqlPredicateFrame<(SqlPredicateAst, Vec<(String, SqlPredicateAst)>)>| {
                match frame {
                    SqlPredicateFrame::Comparison {
                        column,
                        op,
                        literal,
                    } => {
                        let rebuilt =
                            SqlPredicateAst::comparison(column.clone(), op, literal.clone());
                        let mut local_mutants = Vec::new();
                        let flipped_op = match op {
                            SqlComparisonOp::Ge => Some(SqlComparisonOp::Gt),
                            SqlComparisonOp::Le => Some(SqlComparisonOp::Lt),
                            SqlComparisonOp::Gt => Some(SqlComparisonOp::Ge),
                            SqlComparisonOp::Lt => Some(SqlComparisonOp::Le),
                            _ => None,
                        };
                        if let Some(mut_op) = flipped_op {
                            local_mutants.push((
                                "exclusive_boundary".to_string(),
                                SqlPredicateAst::comparison(column, mut_op, literal),
                            ));
                        }
                        (rebuilt, local_mutants)
                    }
                    SqlPredicateFrame::IsNull { column, negated } => {
                        let rebuilt = SqlPredicateAst::is_null(column.clone(), negated);
                        let local_mutants = vec![(
                            "negated_null_check".to_string(),
                            SqlPredicateAst::is_null(column, !negated),
                        )];
                        (rebuilt, local_mutants)
                    }
                    SqlPredicateFrame::InList { column, literals } => {
                        let rebuilt = SqlPredicateAst::in_list(column, literals);
                        (rebuilt, Vec::new())
                    }
                    SqlPredicateFrame::And(
                        (left_ast, left_mutants),
                        (right_ast, right_mutants),
                    ) => {
                        let mut mutants =
                            Vec::with_capacity(3 + left_mutants.len() + right_mutants.len());
                        mutants.push(("dropped_left_conjunct".to_string(), right_ast.clone()));
                        mutants.push(("dropped_right_conjunct".to_string(), left_ast.clone()));
                        mutants.push((
                            "and_to_or".to_string(),
                            SqlPredicateAst::or(left_ast.clone(), right_ast.clone()),
                        ));
                        for (name, m_left) in left_mutants {
                            mutants.push((name, SqlPredicateAst::and(m_left, right_ast.clone())));
                        }
                        for (name, m_right) in right_mutants {
                            mutants.push((name, SqlPredicateAst::and(left_ast.clone(), m_right)));
                        }
                        let rebuilt = SqlPredicateAst::and(left_ast, right_ast);
                        (rebuilt, mutants)
                    }
                    SqlPredicateFrame::Or((left_ast, left_mutants), (right_ast, right_mutants)) => {
                        let mut mutants =
                            Vec::with_capacity(left_mutants.len() + right_mutants.len());
                        for (name, m_left) in left_mutants {
                            mutants.push((name, SqlPredicateAst::or(m_left, right_ast.clone())));
                        }
                        for (name, m_right) in right_mutants {
                            mutants.push((name, SqlPredicateAst::or(left_ast.clone(), m_right)));
                        }
                        let rebuilt = SqlPredicateAst::or(left_ast, right_ast);
                        (rebuilt, mutants)
                    }
                    SqlPredicateFrame::Not((inner_ast, inner_mutants)) => {
                        let mutants = inner_mutants
                            .into_iter()
                            .map(|(name, m_inner)| (name, SqlPredicateAst::not(m_inner)))
                            .collect();
                        let rebuilt = SqlPredicateAst::not(inner_ast);
                        (rebuilt, mutants)
                    }
                }
            },
        );
        mutants
    }

    /// Diagnose a candidate SQL predicate AST against `self.single_bug_mutants()`, distinguishing
    /// between an [`MutantMatchBasis::ExactNormalizedAst`] match and a
    /// [`MutantMatchBasis::SampleRowEquivalence`] match over a non-empty `sample_rows` slice.
    pub fn diagnose_single_bug_mutant(
        &self,
        candidate: &SqlPredicateAst,
        sample_rows: &[BTreeMap<String, serde_json::Value>],
    ) -> Option<MutantMatchDiagnosis> {
        let mutants = self.single_bug_mutants();
        let cand_sql = candidate.render_sql();
        if let Some((name, m)) = mutants
            .iter()
            .find(|(_, m)| m == candidate || m.render_sql() == cand_sql)
        {
            return Some(MutantMatchDiagnosis {
                mutant_name: name.clone(),
                mutant_sql: m.render_sql(),
                match_basis: MutantMatchBasis::ExactNormalizedAst,
            });
        }
        if !sample_rows.is_empty() {
            let self_truth: Vec<bool> = sample_rows.iter().map(|r| self.evaluate_row(r)).collect();
            let cand_truth: Vec<bool> = sample_rows
                .iter()
                .map(|r| candidate.evaluate_row(r))
                .collect();
            if cand_truth != self_truth {
                for (name, m) in &mutants {
                    let m_truth: Vec<bool> =
                        sample_rows.iter().map(|r| m.evaluate_row(r)).collect();
                    if m_truth == cand_truth {
                        return Some(MutantMatchDiagnosis {
                            mutant_name: name.clone(),
                            mutant_sql: m.render_sql(),
                            match_basis: MutantMatchBasis::SampleRowEquivalence {
                                sample_row_count: sample_rows.len(),
                            },
                        });
                    }
                }
            }
        }
        None
    }

    /// Match a candidate SQL predicate AST against `self.single_bug_mutants()` (either by exact
    /// AST equality or by identical row-evaluation truth table over a non-empty `sample_rows` slice).
    pub fn match_single_bug_mutant(
        &self,
        candidate: &SqlPredicateAst,
        sample_rows: &[BTreeMap<String, serde_json::Value>],
    ) -> Option<String> {
        self.diagnose_single_bug_mutant(candidate, sample_rows)
            .map(|d| d.mutant_name)
    }
}
