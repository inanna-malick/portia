//! Structural Rust-AST `todo!("...")` (`!` never-type) seam analyzer (`v2_ternary_hylo::ast_seam`).
//!
//! Classifies cleanroom Rust modules with `todo!("...")` typed holes into the 3-way hylomorphic
//! decomposition lattice (`FORK` `+1` / `ASK` `0` / `INLINE` `-1`) using `syn` + `tree-sitter-rust`.

use crate::rust_ast_lens::{extract_rust_todo_holes, parse_and_validate_rust_cst_and_ast};
use portia_compiler::solvers::clarabel_ternary::CursorPolarity;
use portia_core::{CoreProtocolError, Mode};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Three-way hylomorphic decomposition branch at a Rust AST `todo!("...")` cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HyloTernaryBranch {
    /// `+1.0` on `w_up_tan`: parametrically decoupled sibling `todo!()` holes with verified
    /// non-overlapping mutable borrows, safe for parallel `RoseFrame` fanout.
    Fork,
    /// `0.0` (`|w_up_tan · x_0| <= tol`): underspecified/ambiguous worlds or convex midpoints
    /// between `FORK` and `INLINE` requiring dialogic clarification before code synthesis.
    Ask,
    /// `-1.0` on `w_up_tan`: borrow/mutable-state coupled sequential edits (`&mut self`, `&mut T`,
    /// or overlapping mutable domain state across sibling holes).
    Inline,
}

impl HyloTernaryBranch {
    pub const ALL: [Self; 3] = [Self::Fork, Self::Ask, Self::Inline];

    #[inline]
    #[must_use]
    pub const fn to_cursor_polarity(self) -> CursorPolarity {
        match self {
            Self::Fork => CursorPolarity::PlusOne,
            Self::Ask => CursorPolarity::ZeroNeutral,
            Self::Inline => CursorPolarity::MinusOne,
        }
    }

    #[inline]
    #[must_use]
    pub const fn from_cursor_polarity(polarity: CursorPolarity) -> Self {
        match polarity {
            CursorPolarity::PlusOne => Self::Fork,
            CursorPolarity::ZeroNeutral => Self::Ask,
            CursorPolarity::MinusOne => Self::Inline,
        }
    }

    /// Canonical epistemic [`Mode`] triggered by each ternary hylo branch.
    #[inline]
    #[must_use]
    pub const fn target_mode(self) -> Mode {
        match self {
            Self::Fork => Mode::Write,
            Self::Ask => Mode::Ask,
            Self::Inline => Mode::Read,
        }
    }

    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Fork => "FORK",
            Self::Ask => "ASK",
            Self::Inline => "INLINE",
        }
    }
}

/// Extracted signature & mutability metadata for a single `todo!("...")` (`!` never-type) hole.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeverTypeHoleSeam {
    pub fn_name: String,
    pub intent_message: String,
    pub has_mut_receiver_or_arg: bool,
    pub has_ambiguity_marker: bool,
    pub referenced_domain_types: BTreeSet<String>,
}

/// Complete structural seam classification receipt over a cleanroom Rust module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstSeamClassificationReceipt {
    pub branch: HyloTernaryBranch,
    pub polarity: CursorPolarity,
    pub holes: Vec<NeverTypeHoleSeam>,
    pub mut_coupled_hole_count: usize,
    pub decoupled_hole_count: usize,
    pub shared_mutable_types: BTreeSet<String>,
    pub has_underspecified_invariant: bool,
}

const PRIMITIVE_RUST_TYPES: &[&str] = &[
    "u8", "u16", "u32", "u64", "u128", "usize", "i8", "i16", "i32", "i64", "i128", "isize", "f32",
    "f64", "bool", "char", "str", "String", "Vec", "Option", "Result", "Box", "Slice",
];

fn collect_type_idents(ty: &syn::Type, out: &mut BTreeSet<String>, is_mut_ref: &mut bool) {
    match ty {
        syn::Type::Reference(r) => {
            if r.mutability.is_some() {
                *is_mut_ref = true;
            }
            collect_type_idents(&r.elem, out, is_mut_ref);
        }
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                let ident = seg.ident.to_string();
                if !PRIMITIVE_RUST_TYPES.contains(&ident.as_str()) {
                    out.insert(ident);
                }
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner_ty) = arg {
                            collect_type_idents(inner_ty, out, is_mut_ref);
                        }
                    }
                }
            }
        }
        syn::Type::Slice(s) => collect_type_idents(&s.elem, out, is_mut_ref),
        syn::Type::Tuple(t) => {
            for elem in &t.elems {
                collect_type_idents(elem, out, is_mut_ref);
            }
        }
        _ => {}
    }
}

fn contains_ambiguity_marker(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("ambiguous")
        || lower.contains("underspecified")
        || lower.contains("clarify")
        || lower.contains("unconfirmed")
        || lower.contains("policy_pending")
}

fn inspect_syn_sig(
    sig: &syn::Signature,
    self_ty_ident: Option<&str>,
    intent: &str,
    docs: &[String],
) -> NeverTypeHoleSeam {
    let mut has_mut_receiver_or_arg = false;
    let mut referenced_domain_types = BTreeSet::new();

    for input in &sig.inputs {
        match input {
            syn::FnArg::Receiver(recv) => {
                if recv.reference.is_some() && recv.mutability.is_some() {
                    has_mut_receiver_or_arg = true;
                }
                if let Some(st) = self_ty_ident {
                    referenced_domain_types.insert(st.to_string());
                }
            }
            syn::FnArg::Typed(pat_ty) => {
                let mut arg_mut = false;
                collect_type_idents(&pat_ty.ty, &mut referenced_domain_types, &mut arg_mut);
                if arg_mut {
                    has_mut_receiver_or_arg = true;
                }
            }
        }
    }
    if let syn::ReturnType::Type(_, ret_ty) = &sig.output {
        let mut ret_mut = false;
        collect_type_idents(ret_ty, &mut referenced_domain_types, &mut ret_mut);
        if ret_mut {
            has_mut_receiver_or_arg = true;
        }
    }

    let has_ambiguity_marker =
        contains_ambiguity_marker(intent) || docs.iter().any(|d| contains_ambiguity_marker(d));

    NeverTypeHoleSeam {
        fn_name: sig.ident.to_string(),
        intent_message: intent.to_string(),
        has_mut_receiver_or_arg,
        has_ambiguity_marker,
        referenced_domain_types,
    }
}

fn find_syn_fn_sig<'a>(
    items: &'a [syn::Item],
    fn_name: &str,
) -> Option<(&'a syn::Signature, Option<String>)> {
    for item in items {
        match item {
            syn::Item::Fn(f) if f.sig.ident == fn_name => return Some((&f.sig, None)),
            syn::Item::Impl(imp) => {
                let mut dummy_mut = false;
                let mut self_types = BTreeSet::new();
                collect_type_idents(&imp.self_ty, &mut self_types, &mut dummy_mut);
                let self_ident = self_types.into_iter().next();
                for sub in &imp.items {
                    if let syn::ImplItem::Fn(m) = sub {
                        if m.sig.ident == fn_name {
                            return Some((&m.sig, self_ident));
                        }
                    }
                }
            }
            syn::Item::Mod(m) => {
                if let Some((_, sub_items)) = &m.content {
                    if let Some(found) = find_syn_fn_sig(sub_items, fn_name) {
                        return Some(found);
                    }
                }
            }
            _ => {}
        }
    }
    None
}

/// Inspect all `todo!("...")` (`!` never-type) holes in `rust_source` via `tree-sitter-rust` +
/// `syn::File` and classify the module into `FORK` (`+1`), `ASK` (`0`), or `INLINE` (`-1`).
pub fn classify_rust_todo_seam(
    rust_source: &str,
) -> Result<AstSeamClassificationReceipt, CoreProtocolError> {
    let (_tree, syn_file) = parse_and_validate_rust_cst_and_ast(rust_source)?;
    let raw_holes = extract_rust_todo_holes(rust_source)?;
    if raw_holes.is_empty() {
        return Err(CoreProtocolError::BoundaryViolation(
            "classify_rust_todo_seam requires at least one `todo!(...)` hole".to_string(),
        ));
    }

    let mut holes = Vec::with_capacity(raw_holes.len());
    for h in &raw_holes {
        let (sig, self_ident) = find_syn_fn_sig(&syn_file.items, &h.fn_name).ok_or_else(|| {
            CoreProtocolError::BoundaryViolation(format!(
                "syn signature missing for extracted hole `{}`",
                h.fn_name
            ))
        })?;
        holes.push(inspect_syn_sig(
            sig,
            self_ident.as_deref(),
            &h.intent_message,
            &h.doc_comments,
        ));
    }

    let mut_coupled_hole_count = holes.iter().filter(|h| h.has_mut_receiver_or_arg).count();
    let decoupled_hole_count = holes.len() - mut_coupled_hole_count;
    let has_underspecified_invariant = holes.iter().any(|h| h.has_ambiguity_marker);

    // Detect pairwise overlapping domain types across holes:
    let mut shared_types = BTreeSet::new();
    for i in 0..holes.len() {
        for j in (i + 1)..holes.len() {
            for common in holes[i]
                .referenced_domain_types
                .intersection(&holes[j].referenced_domain_types)
            {
                shared_types.insert(common.clone());
            }
        }
    }
    let shared_mutable_types = if mut_coupled_hole_count > 0 {
        shared_types.clone()
    } else {
        BTreeSet::new()
    };

    // Decision law:
    // 1. Explicit underspecified invariant OR structural midpoint (mixed decoupled + `&mut` coupled
    //    holes in the same module) -> `ASK` (`ZeroNeutral`);
    // 2. All holes borrow `&mut` state or single sequential hole / shared mutable state -> `INLINE` (`MinusOne`);
    // 3. `>= 2` parametrically decoupled holes with disjoint domain types and `0` `&mut` borrows -> `FORK` (`PlusOne`).
    let is_structural_midpoint = mut_coupled_hole_count > 0 && decoupled_hole_count > 0;
    let branch = if has_underspecified_invariant || is_structural_midpoint {
        HyloTernaryBranch::Ask
    } else if mut_coupled_hole_count > 0 || holes.len() < 2 || !shared_types.is_empty() {
        HyloTernaryBranch::Inline
    } else {
        HyloTernaryBranch::Fork
    };

    Ok(AstSeamClassificationReceipt {
        branch,
        polarity: branch.to_cursor_polarity(),
        holes,
        mut_coupled_hole_count,
        decoupled_hole_count,
        shared_mutable_types,
        has_underspecified_invariant,
    })
}
