//! General-purpose `tree-sitter-rust` + `syn` CST/AST validator, module outline formatter,
//! `todo!("...")` hole extractor, and per-function surgical window slicer.

use portia_core::CoreProtocolError;

use super::span::{ByteSpan, ExtractedRustTodoHole};

/// Parsed function/method item inside a Rust source file extracted from the `tree-sitter-rust` CST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedRustFnItem {
    pub(crate) fn_name: String,
    pub(crate) is_test: bool,
    pub(crate) doc_comments: Vec<String>,
    pub(crate) signature: String,
    pub(crate) item_start: usize,
    pub(crate) body_open_brace: usize,
    pub(crate) body_close_brace: usize,
    pub(crate) todo_macro: Option<(ByteSpan, String)>,
}

/// Parse and validate a Rust source file using both `tree-sitter-rust` (`!root_node().has_error()`)
/// and `syn::parse_file`.
pub fn parse_and_validate_rust_cst_and_ast(
    source: &str,
) -> Result<(tree_sitter::Tree, syn::File), CoreProtocolError> {
    let mut parser = tree_sitter::Parser::new();
    let language: tree_sitter::Language = tree_sitter_rust::LANGUAGE.into();
    parser.set_language(&language).map_err(|e| {
        CoreProtocolError::BoundaryViolation(format!(
            "failed to load tree-sitter-rust grammar: {e}"
        ))
    })?;
    let tree = parser.parse(source, None).ok_or_else(|| {
        CoreProtocolError::BoundaryViolation("tree-sitter-rust returned None on parse".to_string())
    })?;
    if tree.root_node().has_error() {
        return Err(CoreProtocolError::BoundaryViolation(
            "tree-sitter-rust CST contains syntax error nodes".to_string(),
        ));
    }
    let syn_file = syn::parse_file(source).map_err(|e| {
        CoreProtocolError::BoundaryViolation(format!("syn::parse_file syntax error: {e}"))
    })?;
    Ok((tree, syn_file))
}

/// Verify that a Rust module parses cleanly under both `tree-sitter-rust` and `syn::parse_file`.
pub fn verify_balanced_rust_delimiters(source: &str) -> Result<(), CoreProtocolError> {
    let _ = parse_and_validate_rust_cst_and_ast(source)?;
    Ok(())
}

/// Recursively search a `tree-sitter-rust` `block` node for the first `macro_invocation` whose
/// macro path is `todo`, returning `(ByteSpan, intent_message)`.
fn find_todo_macro_in_cst_node(
    node: tree_sitter::Node<'_>,
    source: &str,
) -> Option<(ByteSpan, String)> {
    if node.kind() == "macro_invocation" {
        let macro_ident_node = node
            .child_by_field_name("macro")
            .or_else(|| node.named_child(0));
        if let Some(m_node) = macro_ident_node {
            let macro_name = source[m_node.byte_range()].trim();
            if macro_name == "todo" || macro_name.ends_with("::todo") {
                let replace_end = match node.parent() {
                    Some(parent) if parent.kind() == "expression_statement" => parent.end_byte(),
                    _ => node.end_byte(),
                };
                let mut intent = String::new();
                for i in 0..node.named_child_count() {
                    if let Some(child) = node.named_child(i) {
                        if child.kind() == "token_tree" {
                            let raw_tt = source[child.byte_range()].trim();
                            let inner = raw_tt
                                .strip_prefix('(')
                                .and_then(|s| s.strip_suffix(')'))
                                .or_else(|| {
                                    raw_tt.strip_prefix('[').and_then(|s| s.strip_suffix(']'))
                                })
                                .or_else(|| {
                                    raw_tt.strip_prefix('{').and_then(|s| s.strip_suffix('}'))
                                })
                                .unwrap_or(raw_tt)
                                .trim();
                            intent = inner
                                .strip_prefix('"')
                                .and_then(|s| s.strip_suffix('"'))
                                .unwrap_or(inner)
                                .to_string();
                            break;
                        }
                    }
                }
                let span = ByteSpan::try_new(node.start_byte(), replace_end).ok()?;
                return Some((span, intent));
            }
        }
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            if let Some(found) = find_todo_macro_in_cst_node(child, source) {
                return Some(found);
            }
        }
    }
    None
}

/// Collect concrete `function_item` nodes (with `{ ... }` bodies) from a `tree-sitter-rust` CST node
/// in source order, including functions inside `impl_item` and `mod_item` blocks.
fn collect_rust_fn_items_from_cst(
    node: tree_sitter::Node<'_>,
    rust_source: &str,
    out: &mut Vec<ParsedRustFnItem>,
) {
    if node.kind() == "function_item" {
        let Some(name_node) = node.child_by_field_name("name") else {
            return;
        };
        let Some(body_node) = node.child_by_field_name("body") else {
            return;
        };
        let fn_name = rust_source[name_node.byte_range()].trim().to_string();
        let mut item_start = node.start_byte();
        let mut is_test = false;
        let mut leading_docs_rev = Vec::new();

        let mut prev = node.prev_sibling();
        while let Some(sib) = prev {
            match sib.kind() {
                "attribute_item" => {
                    let attr_txt = rust_source[sib.byte_range()].trim();
                    if attr_txt.contains("test") {
                        is_test = true;
                    }
                    item_start = sib.start_byte();
                    leading_docs_rev.push(attr_txt.to_string());
                    prev = sib.prev_sibling();
                }
                "line_comment" | "block_comment" => {
                    let comment_txt = rust_source[sib.byte_range()].trim();
                    if comment_txt.starts_with("///")
                        || comment_txt.starts_with("//!")
                        || comment_txt.starts_with("/**")
                    {
                        item_start = sib.start_byte();
                        leading_docs_rev.push(comment_txt.to_string());
                        prev = sib.prev_sibling();
                    } else {
                        break;
                    }
                }
                _ => break,
            }
        }
        leading_docs_rev.reverse();

        let open_brace = body_node.start_byte();
        let close_brace = body_node.end_byte().saturating_sub(1);
        let sig_raw = rust_source[node.start_byte()..open_brace].trim();
        let normalized_sig = sig_raw.split_whitespace().collect::<Vec<_>>().join(" ");
        let todo_macro = find_todo_macro_in_cst_node(body_node, rust_source);

        out.push(ParsedRustFnItem {
            fn_name,
            is_test,
            doc_comments: leading_docs_rev,
            signature: normalized_sig,
            item_start,
            body_open_brace: open_brace,
            body_close_brace: close_brace,
            todo_macro,
        });
        return;
    }

    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            if matches!(
                child.kind(),
                "function_item" | "impl_item" | "mod_item" | "declaration_list"
            ) {
                collect_rust_fn_items_from_cst(child, rust_source, out);
            }
        }
    }
}

/// Parse all concrete function definitions (with `{ ... }` bodies) from a Rust source string
/// using `tree-sitter-rust` CST traversal and `syn::parse_file` verification.
pub(crate) fn parse_rust_fn_items(
    rust_source: &str,
) -> Result<Vec<ParsedRustFnItem>, CoreProtocolError> {
    let (tree, _syn_file) = parse_and_validate_rust_cst_and_ast(rust_source)?;
    let mut items = Vec::new();
    collect_rust_fn_items_from_cst(tree.root_node(), rust_source, &mut items);
    Ok(items)
}

/// Format a compact, rule-free `[AST ...]` header from a Rust module (`struct`, `enum`, `trait`,
/// `fn`, `#[test] fn`) using `tree-sitter-rust` CST nodes.
pub fn format_rust_ast_outline(
    module_name: &str,
    rust_source: &str,
) -> Result<String, CoreProtocolError> {
    let (tree, _syn_file) = parse_and_validate_rust_cst_and_ast(rust_source)?;
    let root = tree.root_node();

    let mut lines = Vec::new();
    lines.push(format!("[Module `{module_name}`]"));

    for i in 0..root.named_child_count() {
        let Some(node) = root.named_child(i) else {
            continue;
        };
        match node.kind() {
            "struct_item" => {
                let Some(name_node) = node.child_by_field_name("name") else {
                    continue;
                };
                let struct_name = rust_source[name_node.byte_range()].trim();
                let mut fields = Vec::new();
                if let Some(body_node) = node.child_by_field_name("body") {
                    for j in 0..body_node.named_child_count() {
                        if let Some(field_node) = body_node.named_child(j) {
                            if field_node.kind() == "field_declaration" {
                                if let (Some(fname), Some(ftype)) = (
                                    field_node.child_by_field_name("name"),
                                    field_node.child_by_field_name("type"),
                                ) {
                                    fields.push(format!(
                                        "`{}: {}`",
                                        rust_source[fname.byte_range()].trim(),
                                        rust_source[ftype.byte_range()].trim()
                                    ));
                                }
                            }
                        }
                    }
                }
                lines.push(format!("[Struct `{struct_name}` ({})]", fields.join(", ")));
            }
            "enum_item" => {
                let Some(name_node) = node.child_by_field_name("name") else {
                    continue;
                };
                let enum_name = rust_source[name_node.byte_range()].trim();
                let mut variants = Vec::new();
                if let Some(body_node) = node.child_by_field_name("body") {
                    for j in 0..body_node.named_child_count() {
                        if let Some(var_node) = body_node.named_child(j) {
                            if var_node.kind() == "enum_variant" {
                                let v_txt = rust_source[var_node.byte_range()]
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                variants.push(format!("`{v_txt}`"));
                            }
                        }
                    }
                }
                lines.push(format!("[Enum `{enum_name}` ({})]", variants.join(", ")));
            }
            "trait_item" => {
                let Some(name_node) = node.child_by_field_name("name") else {
                    continue;
                };
                let trait_name = rust_source[name_node.byte_range()].trim();
                let mut methods = Vec::new();
                if let Some(body_node) = node.child_by_field_name("body") {
                    for j in 0..body_node.named_child_count() {
                        if let Some(m_node) = body_node.named_child(j) {
                            if m_node.kind() == "function_signature_item"
                                || m_node.kind() == "function_item"
                            {
                                let m_raw = rust_source[m_node.byte_range()]
                                    .trim()
                                    .trim_end_matches(';');
                                let m_clean = m_raw
                                    .split('{')
                                    .next()
                                    .unwrap_or(m_raw)
                                    .split_whitespace()
                                    .collect::<Vec<_>>()
                                    .join(" ");
                                methods.push(format!("`{m_clean}`"));
                            }
                        }
                    }
                }
                lines.push(format!("[Trait `{trait_name}` ({})]", methods.join(", ")));
            }
            _ => {}
        }
    }

    let mut fn_items = Vec::new();
    collect_rust_fn_items_from_cst(root, rust_source, &mut fn_items);
    let mut hole_counter = 0usize;
    for item in fn_items {
        let sig_after_name = item
            .signature
            .find(&item.fn_name)
            .map(|pos| item.signature[pos + item.fn_name.len()..].trim())
            .unwrap_or("");
        let tag = if item.is_test { "TestFn" } else { "Fn" };
        if let Some((_span, intent)) = item.todo_macro {
            let idx = hole_counter;
            hole_counter += 1;
            lines.push(format!(
                "[{tag} `{}`{} [TODO_HOLE #{idx}: \"{intent}\"]]",
                item.fn_name, sig_after_name
            ));
        } else {
            lines.push(format!(
                "[{tag} `{}`{} [IMPLEMENTED]]",
                item.fn_name, sig_after_name
            ));
        }
    }

    Ok(lines.join("\n"))
}

/// Locate every function/method/test body containing `todo!("...")` with exact byte spans and
/// `tree-sitter-rust` + `syn` verification.
pub fn extract_rust_todo_holes(
    rust_source: &str,
) -> Result<Vec<ExtractedRustTodoHole>, CoreProtocolError> {
    let fn_items = parse_rust_fn_items(rust_source)?;
    let mut holes = Vec::new();
    for item in fn_items {
        if let Some((todo_macro_span, intent_message)) = item.todo_macro {
            let fn_span = ByteSpan::try_new(item.item_start, item.body_close_brace + 1)?;
            let hole_index = holes.len();
            holes.push(ExtractedRustTodoHole {
                hole_index,
                fn_name: item.fn_name,
                doc_comments: item.doc_comments,
                signature: item.signature,
                todo_macro_span,
                fn_span,
                intent_message,
            });
        }
    }
    Ok(holes)
}

/// Return the $O(1)$ surgical window for `Mode::Read` on a single function (`doc_comments` +
/// `signature` + current `{ ... }` body) while eliding all other function bodies in the module.
pub fn slice_rust_fn_window(
    rust_source: &str,
    target_fn_name: &str,
) -> Result<String, CoreProtocolError> {
    let clean_target = target_fn_name.trim().trim_matches('`');
    let fn_items = parse_rust_fn_items(rust_source)?;
    let item = fn_items
        .into_iter()
        .find(|f| f.fn_name == clean_target)
        .ok_or_else(|| {
            CoreProtocolError::BoundaryViolation(format!(
                "function `{clean_target}` not found in Rust module"
            ))
        })?;

    Ok(rust_source[item.item_start..=item.body_close_brace]
        .trim()
        .to_string())
}
