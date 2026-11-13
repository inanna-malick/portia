use crate::{
    CheckpointPatchPlan, ColFill, ConfigIntermediateSizeUpdate, MlpWidenPlan, RowFill,
    SurgeonError, TensorEdit,
};
use std::collections::BTreeSet;

pub(crate) fn lower(spec: MlpWidenPlan) -> Result<CheckpointPatchPlan, SurgeonError> {
    if spec.num_layers == 0
        || spec.hidden_size == 0
        || spec.base_intermediate_size == 0
        || spec.added_slots == 0
        || spec.layer_prefix.is_empty()
        || spec.layer_prefix.ends_with('.')
    {
        return Err(SurgeonError::ConfigMismatch(
            "MLP dimensions and layer count must be positive, with a nonempty layer prefix without a trailing dot"
                .into(),
        ));
    }
    let new_size = spec
        .base_intermediate_size
        .checked_add(spec.added_slots)
        .ok_or_else(|| invalid_layout("MLP intermediate size overflow"))?;
    let length = spec
        .added_slots
        .checked_mul(spec.hidden_size)
        .ok_or_else(|| invalid_layout("MLP append buffer length overflow"))?;
    new_size
        .checked_mul(spec.hidden_size)
        .ok_or_else(|| invalid_layout("MLP output element count overflow"))?;

    let mut targets = BTreeSet::new();
    for layer in &spec.target_layers {
        if layer.layer_index >= spec.num_layers {
            return Err(SurgeonError::ConfigMismatch(format!(
                "MLP layer {} is outside 0..{}",
                layer.layer_index, spec.num_layers
            )));
        }
        if !targets.insert(layer.layer_index) {
            return Err(SurgeonError::DuplicateTensorTarget {
                name: tensor_name(&spec.layer_prefix, layer.layer_index, "gate_proj"),
            });
        }
    }
    let mut result = CheckpointPatchPlan {
        output_strategy: spec.output_strategy,
        config_intermediate_size: spec.uniform_across_all_layers.then_some(
            ConfigIntermediateSizeUpdate {
                expected_old_size: spec.base_intermediate_size,
                new_size,
            },
        ),
        ..CheckpointPatchPlan::default()
    };
    for layer in spec.target_layers {
        for (projection, data, is_down) in [
            ("gate_proj", layer.gate_rows, false),
            ("up_proj", layer.up_rows, false),
            ("down_proj", layer.down_cols, true),
        ] {
            let name = tensor_name(&spec.layer_prefix, layer.layer_index, projection);
            if data.len() != length {
                return Err(SurgeonError::BufferLengthMismatch {
                    name,
                    expected: length,
                    actual: data.len(),
                });
            }
            if let Some(index) = data.iter().position(|value| !value.is_finite()) {
                let (row, col) = if is_down {
                    (
                        index / spec.added_slots,
                        spec.base_intermediate_size + index % spec.added_slots,
                    )
                } else {
                    (
                        spec.base_intermediate_size + index / spec.hidden_size,
                        index % spec.hidden_size,
                    )
                };
                return Err(SurgeonError::NonFiniteResult { name, row, col });
            }
            let (shape, edit) = if is_down {
                (
                    (spec.hidden_size, spec.base_intermediate_size),
                    TensorEdit::AppendCols2D {
                        fill: ColFill::Values {
                            added_cols: spec.added_slots,
                            data,
                        },
                    },
                )
            } else {
                (
                    (spec.base_intermediate_size, spec.hidden_size),
                    TensorEdit::AppendRows2D {
                        fill: RowFill::Values {
                            added_rows: spec.added_slots,
                            data,
                        },
                    },
                )
            };
            result.expected_shapes.insert(name.clone(), shape);
            result.insert_edit(name, edit)?;
        }
    }
    if spec.uniform_across_all_layers {
        for layer in (0..spec.num_layers).filter(|layer| !targets.contains(layer)) {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let name = tensor_name(&spec.layer_prefix, layer, projection);
                let (shape, edit) = if projection == "down_proj" {
                    (
                        (spec.hidden_size, spec.base_intermediate_size),
                        TensorEdit::AppendCols2D {
                            fill: ColFill::Zeros {
                                added_cols: spec.added_slots,
                            },
                        },
                    )
                } else {
                    (
                        (spec.base_intermediate_size, spec.hidden_size),
                        TensorEdit::AppendRows2D {
                            fill: RowFill::Zeros {
                                added_rows: spec.added_slots,
                            },
                        },
                    )
                };
                result.expected_shapes.insert(name.clone(), shape);
                result.insert_edit(name, edit)?;
            }
        }
    }
    Ok(result)
}

fn tensor_name(prefix: &str, layer: usize, projection: &str) -> String {
    format!("{prefix}.{layer}.mlp.{projection}.weight")
}

fn invalid_layout(message: &str) -> SurgeonError {
    SurgeonError::SafetensorsFormat {
        path: "<MLP plan>".into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LayerMlpAppendSpec, OutputStrategy};

    fn spec() -> MlpWidenPlan {
        MlpWidenPlan {
            layer_prefix: "model.layers".into(),
            num_layers: 3,
            hidden_size: 2,
            base_intermediate_size: 4,
            added_slots: 1,
            target_layers: vec![LayerMlpAppendSpec {
                layer_index: 1,
                gate_rows: vec![1.0, 2.0],
                up_rows: vec![3.0, 4.0],
                down_cols: vec![5.0, 6.0],
            }],
            uniform_across_all_layers: true,
            output_strategy: OutputStrategy::HardlinkUnmodifiedShards,
        }
    }

    #[test]
    fn uniform_lowering_preserves_values_and_all_expected_shapes() {
        let result = spec().into_patch_plan().unwrap();
        assert_eq!(result.edits.len(), 9);
        assert_eq!(result.expected_shapes.len(), 9);
        assert_eq!(
            result.output_strategy,
            OutputStrategy::HardlinkUnmodifiedShards
        );
        assert_eq!(
            result.config_intermediate_size,
            Some(ConfigIntermediateSizeUpdate {
                expected_old_size: 4,
                new_size: 5
            })
        );
        for layer in 0..3 {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                let name = tensor_name("model.layers", layer, projection);
                assert_eq!(
                    result.expected_shapes[&name],
                    if projection == "down_proj" {
                        (2, 4)
                    } else {
                        (4, 2)
                    }
                );
            }
        }
        assert_eq!(
            result.edits["model.layers.1.mlp.down_proj.weight"],
            TensorEdit::AppendCols2D {
                fill: ColFill::Values {
                    added_cols: 1,
                    data: vec![5.0, 6.0]
                }
            }
        );
        assert_eq!(
            result.edits["model.layers.0.mlp.gate_proj.weight"],
            TensorEdit::AppendRows2D {
                fill: RowFill::Zeros { added_rows: 1 }
            }
        );
    }

    #[test]
    fn nonuniform_lowering_keeps_shape_guards_without_config_update() {
        let mut input = spec();
        input.uniform_across_all_layers = false;
        let result = input.into_patch_plan().unwrap();
        assert_eq!(result.edits.len(), 3);
        assert_eq!(result.expected_shapes.len(), 3);
        assert!(result.config_intermediate_size.is_none());
    }

    #[test]
    fn invalid_parameters_buffers_duplicates_and_overflow_reject() {
        let mut input = spec();
        input.added_slots = 0;
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::ConfigMismatch(_))
        ));
        let mut input = spec();
        input.target_layers[0].layer_index = 3;
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::ConfigMismatch(_))
        ));
        let mut input = spec();
        input.target_layers.push(input.target_layers[0].clone());
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::DuplicateTensorTarget { .. })
        ));
        let mut input = spec();
        input.target_layers[0].gate_rows.pop();
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::BufferLengthMismatch {
                expected: 2,
                actual: 1,
                ..
            })
        ));
        let mut input = spec();
        input.base_intermediate_size = usize::MAX;
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::SafetensorsFormat { .. })
        ));
    }

    #[test]
    fn nonfinite_coordinates_refer_to_output_tensor() {
        let mut input = spec();
        input.target_layers[0].up_rows[1] = f32::NAN;
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::NonFiniteResult { row: 4, col: 1, .. })
        ));
        let mut input = spec();
        input.target_layers[0].down_cols[1] = f32::INFINITY;
        assert!(matches!(
            input.into_patch_plan(),
            Err(SurgeonError::NonFiniteResult { row: 1, col: 4, .. })
        ));
    }

    #[test]
    fn checked_insert_keeps_the_first_edit_on_duplicate() {
        let mut plan = CheckpointPatchPlan::default();
        let edit = TensorEdit::AppendRows2D {
            fill: RowFill::Zeros { added_rows: 1 },
        };
        plan.insert_edit("weight", edit.clone()).unwrap();
        assert!(matches!(
            plan.insert_edit(
                "weight",
                TensorEdit::AppendRows2D {
                    fill: RowFill::Zeros { added_rows: 3 }
                }
            ),
            Err(SurgeonError::DuplicateTensorTarget { .. })
        ));
        assert_eq!(plan.edits["weight"], edit);
    }
}
