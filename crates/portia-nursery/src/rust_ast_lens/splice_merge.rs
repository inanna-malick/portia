//! Exoskeleton-safe `todo!("...")` leaf hole splicer and hylomorphic octo-merger.

use portia_core::hylo::{try_expand_and_collapse, PartiallyApplied, RoseFrame};
use portia_core::CoreProtocolError;

use super::cst_extract::{
    extract_rust_todo_holes, parse_and_validate_rust_cst_and_ast, parse_rust_fn_items,
};
use super::span::ExtractedRustTodoHole;

/// Validate a candidate hole replacement body via `syn::parse_str::<syn::Block>` and verify that
/// its AST statements contain zero forbidden top-level `syn::Item` declarations (`struct`, `enum`,
/// `trait`, `fn`, `impl`, `mod`) and zero un-filled `todo!(...)` macros.
pub(crate) fn validate_hole_replacement_block_ast(
    target_fn_name: &str,
    body_inner: &str,
) -> Result<(), CoreProtocolError> {
    use syn::visit::Visit;

    struct TodoMacroVisitor {
        found_todo: bool,
    }

    impl<'ast> Visit<'ast> for TodoMacroVisitor {
        fn visit_macro(&mut self, i: &'ast syn::Macro) {
            if let Some(last_seg) = i.path.segments.last() {
                if last_seg.ident == "todo" {
                    self.found_todo = true;
                }
            }
            syn::visit::visit_macro(self, i);
        }
    }

    let wrapped_block = format!("{{ {body_inner} }}");
    let parsed_block = syn::parse_str::<syn::Block>(&wrapped_block).map_err(|e| {
        CoreProtocolError::BoundaryViolation(format!(
            "replacement body for `{target_fn_name}` failed syn::Block parse: {e}"
        ))
    })?;
    let mut todo_visitor = TodoMacroVisitor { found_todo: false };
    todo_visitor.visit_block(&parsed_block);
    if todo_visitor.found_todo {
        return Err(CoreProtocolError::BoundaryViolation(format!(
            "replacement body for `{target_fn_name}` still contains an un-filled `todo!(...)` macro"
        )));
    }
    for stmt in &parsed_block.stmts {
        if let syn::Stmt::Item(item) = stmt {
            let forbidden_kind = match item {
                syn::Item::Struct(_) => Some("struct"),
                syn::Item::Enum(_) => Some("enum"),
                syn::Item::Trait(_) => Some("trait"),
                syn::Item::Fn(_) => Some("fn"),
                syn::Item::Impl(_) => Some("impl"),
                syn::Item::Mod(_) => Some("mod"),
                _ => None,
            };
            if let Some(kw) = forbidden_kind {
                return Err(CoreProtocolError::BoundaryViolation(format!(
                    "exoskeleton violation: leaf hole patch for `{target_fn_name}` attempted top-level `{kw}` item injection"
                )));
            }
        }
    }
    Ok(())
}

/// Exoskeleton-safe leaf hole splicer: replaces only the `todo!("...")` macro
/// invocation (or block body) inside `target_fn_name`, validates `new_body_inner` via `syn::Block`,
/// rejects top-level `syn::Item` injection outside the hole, and verifies the resulting file with
/// both `tree-sitter-rust` and `syn::parse_file`.
pub fn splice_rust_todo_hole_body(
    rust_source: &str,
    target_fn_name: &str,
    new_body_inner: &str,
) -> Result<String, CoreProtocolError> {
    let trimmed_body = new_body_inner.trim();
    if trimmed_body.is_empty() {
        return Err(CoreProtocolError::BoundaryViolation(
            "replacement body for todo! hole cannot be empty".to_string(),
        ));
    }
    validate_hole_replacement_block_ast(target_fn_name, trimmed_body)?;

    let clean_target = target_fn_name.trim().trim_matches('`');
    let fn_items = parse_rust_fn_items(rust_source)?;
    let item = fn_items
        .into_iter()
        .find(|f| f.fn_name == clean_target)
        .ok_or_else(|| {
            CoreProtocolError::BoundaryViolation(format!(
                "target function `{clean_target}` not found in module"
            ))
        })?;

    let (replace_start, replace_end) = if let Some((todo_span, _)) = item.todo_macro {
        (todo_span.start(), todo_span.end())
    } else {
        (item.body_open_brace + 1, item.body_close_brace)
    };

    let mut spliced = String::with_capacity(
        rust_source.len() - (replace_end - replace_start) + trimmed_body.len() + 16,
    );
    spliced.push_str(&rust_source[..replace_start]);
    if item.todo_macro.is_some() {
        spliced.push_str(trimmed_body);
    } else {
        spliced.push_str("\n        ");
        spliced.push_str(trimmed_body);
        spliced.push_str("\n    ");
    }
    spliced.push_str(&rust_source[replace_end..]);

    let _ = parse_and_validate_rust_cst_and_ast(&spliced)?;
    Ok(spliced)
}

/// Use [`portia_core::hylo::try_expand_and_collapse`] (`RoseFrame`) to unfold `skeleton_source` into its
/// [`ExtractedRustTodoHole`] leaves, validate each `(fn_name, body_patch)` in isolation, and
/// collapse/splice all verified leaf patches back into `skeleton_source`.
pub fn hylo_octo_merge_rust_holes(
    skeleton_source: &str,
    leaf_patches: &[(String, String)],
) -> Result<String, CoreProtocolError> {
    #[derive(Clone)]
    enum OctoSeed {
        Root {
            skeleton: String,
            patches: Vec<(String, String)>,
        },
        Leaf {
            hole: ExtractedRustTodoHole,
            patch_body: String,
        },
    }

    #[derive(Clone)]
    enum OctoCollapsed {
        Skeleton(String),
        VerifiedPatch { fn_name: String, body: String },
    }

    let out = try_expand_and_collapse::<
        RoseFrame<OctoCollapsed, PartiallyApplied>,
        OctoSeed,
        OctoCollapsed,
        CoreProtocolError,
    >(
        OctoSeed::Root {
            skeleton: skeleton_source.to_string(),
            patches: leaf_patches.to_vec(),
        },
        |seed| match seed {
            OctoSeed::Root { skeleton, patches } => {
                let holes = extract_rust_todo_holes(&skeleton)?;
                let mut children = Vec::with_capacity(patches.len());
                for (fn_name, patch_body) in patches {
                    let clean_name = fn_name.trim().trim_matches('`');
                    let hole = holes
                        .iter()
                        .find(|h| h.fn_name == clean_name)
                        .cloned()
                        .ok_or_else(|| {
                            CoreProtocolError::BoundaryViolation(format!(
                                "octo-merge target hole `{clean_name}` not found in skeleton"
                            ))
                        })?;
                    children.push(OctoSeed::Leaf { hole, patch_body });
                }
                Ok(RoseFrame {
                    label: OctoCollapsed::Skeleton(skeleton),
                    children,
                })
            }
            OctoSeed::Leaf { hole, patch_body } => {
                let _isolated =
                    splice_rust_todo_hole_body(skeleton_source, &hole.fn_name, &patch_body)?;
                Ok(RoseFrame {
                    label: OctoCollapsed::VerifiedPatch {
                        fn_name: hole.fn_name,
                        body: patch_body,
                    },
                    children: Vec::new(),
                })
            }
        },
        |frame| match frame.label {
            OctoCollapsed::VerifiedPatch { fn_name, body } => {
                Ok(OctoCollapsed::VerifiedPatch { fn_name, body })
            }
            OctoCollapsed::Skeleton(mut current_source) => {
                for child in frame.children {
                    if let OctoCollapsed::VerifiedPatch { fn_name, body } = child {
                        current_source =
                            splice_rust_todo_hole_body(&current_source, &fn_name, &body)?;
                    }
                }
                Ok(OctoCollapsed::Skeleton(current_source))
            }
        },
    )?;

    match out {
        OctoCollapsed::Skeleton(merged) => Ok(merged),
        OctoCollapsed::VerifiedPatch { body, .. } => Ok(body),
    }
}
