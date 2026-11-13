//! Primary-table schema layout, surface-isomorphism DDL, and batched haystack row synthesis.

use portia_core::domain::{LockState, PolicyState};
use portia_nursery::nursery_scenario::UnifiedScenarioTrapModifier;
use portia_nursery::roguelite_lattice::{
    generate_compound_policy_matrix, generate_identifier_swarm, PolicyLockSurfaceIso,
    PrimaryKeyIsomorphism,
};

use super::super::schema_gen::build_shared_column_layout;
use super::secondary_ddl::append_trap_rows_and_secondary_tables;
use super::spec::{UnifiedSqlScenarioInstance, UnifiedSqlScenarioSpec};

pub(super) struct PrimaryTableRowFormatter<'a> {
    pub(super) spec: &'a UnifiedSqlScenarioSpec,
    pub(super) all_primary_columns: &'a [String],
    pub(super) deceptive_decoy_columns: &'a [String],
    pub(super) homonym_lookalike_columns: &'a [String],
    pub(super) composite_tenant_code: Option<&'a str>,
    pub(super) is_numeric_tier_iso: bool,
}

impl PrimaryTableRowFormatter<'_> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn format_tuple(
        &self,
        pk: i64,
        slug: &str,
        bal: i64,
        pol_state: PolicyState,
        lck_state: LockState,
        steward: &str,
        lifecycle: &str,
        guard_vec: &[i64],
    ) -> String {
        let pol_str = self.spec.policy_lock_iso.encode_policy(pol_state);
        let lck_str = self.spec.policy_lock_iso.encode_lock(lck_state);
        let (pol_bit, _) = self
            .spec
            .policy_lock_iso
            .encode_split_policy_bits(pol_state);
        let (lck_bit, _) = self.spec.policy_lock_iso.encode_split_lock_bits(lck_state);

        let vals: Vec<String> = self
            .all_primary_columns
            .iter()
            .map(|col_name| {
                if col_name == "record_id" {
                    pk.to_string()
                } else if col_name == "tenant_code" {
                    let t_code = self.composite_tenant_code.unwrap_or("DEFAULT_TENANT");
                    format!("'{t_code}'")
                } else if col_name == "entity_slug" {
                    format!("'{slug}'")
                } else if col_name == "balance_credits" {
                    bal.to_string()
                } else if col_name == "policy_decision" {
                    if self.is_numeric_tier_iso {
                        pol_str.clone()
                    } else {
                        format!("'{pol_str}'")
                    }
                } else if col_name == "lock_state" {
                    if self.is_numeric_tier_iso {
                        lck_str.clone()
                    } else {
                        format!("'{lck_str}'")
                    }
                } else if col_name == "steward_handle" {
                    format!("'{steward}'")
                } else if col_name == "lifecycle_status" {
                    format!("'{lifecycle}'")
                } else if col_name == "policy_tier_bit" {
                    pol_bit.to_string()
                } else if col_name == "lock_hold_bit" {
                    lck_bit.to_string()
                } else if let Some(g_idx_str) = col_name.strip_prefix("policy_guard_c") {
                    let g_idx = g_idx_str.parse::<usize>().unwrap_or(0);
                    guard_vec.get(g_idx).copied().unwrap_or(0).to_string()
                } else if self.deceptive_decoy_columns.iter().any(|d| d == col_name) {
                    if col_name.contains("balance")
                        || col_name.contains("quota")
                        || col_name.contains("timer")
                    {
                        "999999".to_string()
                    } else {
                        "'DEPRECATED_TRAP'".to_string()
                    }
                } else if self.homonym_lookalike_columns.iter().any(|h| h == col_name) {
                    format!("'HOMONYM_SHADOW_{slug}'")
                } else {
                    "'TELEMETRY_OK'".to_string()
                }
            })
            .collect();
        format!("({})", vals.join(", "))
    }
}

impl UnifiedSqlScenarioSpec {
    /// Compile the unified tensor-product SQLite DDL and data rows across all SQL/swarm/policy/nursery dimensions.
    pub fn build(&self) -> UnifiedSqlScenarioInstance {
        let total_rows = self.stress_vector.row_haystack_size.clamp(1, 1000);
        let needle_idx = self.stress_vector.needle_depth.resolve_index(total_rows);
        let needle_pk = self.primary_key_iso.resolve_row_pk(needle_idx);

        let swarm_count = self
            .stress_vector
            .identifier_swarm_collision
            .clamp(1, total_rows.min(250));
        let swarm = generate_identifier_swarm(
            "acct-7f3a-07a",
            swarm_count,
            self.stress_vector.collision_locus,
            self.seed,
        );
        let needle_slug = swarm.target_identifier.clone();
        let needle_surface_key = self
            .primary_key_iso
            .format_surface_key(needle_idx, &needle_slug);

        let policy_arity = self.stress_vector.policy_predicate_arity.clamp(1, 16);
        let policy_matrix = generate_compound_policy_matrix(policy_arity, self.seed);
        let policy_guard_columns: Vec<String> = policy_matrix
            .clauses
            .iter()
            .map(|c| c.column_name.clone())
            .collect();

        let has_arch_m2 = self
            .trap_modifiers
            .contains(&UnifiedScenarioTrapModifier::ArchDecoyHoneypotM2);
        let has_ambiguous_cand = self
            .trap_modifiers
            .contains(&UnifiedScenarioTrapModifier::AmbiguousEntityCandidateDisambiguation);
        let has_split_bool = self.policy_lock_iso == PolicyLockSurfaceIso::SplitTwoColumnBoolean;
        let is_numeric_tier_iso = self.policy_lock_iso == PolicyLockSurfaceIso::NumericTierBoolean;
        let composite_tenant_code: Option<String> = match &self.primary_key_iso {
            PrimaryKeyIsomorphism::CompositeTenantKey { tenant_code, .. } => {
                Some(tenant_code.clone())
            }
            _ => None,
        };
        let is_uuid_slug_pk = matches!(
            self.primary_key_iso,
            PrimaryKeyIsomorphism::SwarmUuidSlugPrimaryKey
        );
        let underflow_floor: Option<i64> = self.trap_modifiers.iter().find_map(|m| match m {
            UnifiedScenarioTrapModifier::ConservationUnderflowFloorH4 { min_floor, .. } => {
                Some(*min_floor)
            }
            _ => None,
        });

        let (
            mut canonical_columns,
            deceptive_decoy_columns,
            raw_noise_columns,
            mut all_primary_columns,
        ) = build_shared_column_layout(
            "record_id",
            self.stress_vector.column_width_arity,
            self.stress_vector.deprecated_decoy_columns,
            &policy_guard_columns,
        );

        let requested_homonyms = self.stress_vector.homonym_overload_count.clamp(1, 12);
        let homonym_count = requested_homonyms.min(raw_noise_columns.len());
        let mut homonym_lookalike_columns = Vec::with_capacity(homonym_count);
        let mut noise_telemetry_columns = Vec::with_capacity(raw_noise_columns.len());
        for (idx, old_noise) in raw_noise_columns.iter().enumerate() {
            if idx < homonym_count {
                let h_col = format!("homonym_overload_attr_{:03}", idx + 1);
                homonym_lookalike_columns.push(h_col.clone());
                noise_telemetry_columns.push(h_col.clone());
                if let Some(pos) = all_primary_columns.iter().position(|c| c == old_noise) {
                    all_primary_columns[pos] = h_col;
                }
            } else {
                noise_telemetry_columns.push(old_noise.clone());
            }
        }

        if composite_tenant_code.is_some()
            && !all_primary_columns.iter().any(|c| c == "tenant_code")
        {
            canonical_columns.push("tenant_code".to_string());
            all_primary_columns.push("tenant_code".to_string());
        }
        if has_arch_m2 && !all_primary_columns.iter().any(|c| c == "lifecycle_status") {
            canonical_columns.push("lifecycle_status".to_string());
            all_primary_columns.push("lifecycle_status".to_string());
        }
        if has_split_bool {
            for split_col in ["policy_tier_bit", "lock_hold_bit"] {
                if !all_primary_columns.iter().any(|c| c == split_col) {
                    canonical_columns.push(split_col.to_string());
                    all_primary_columns.push(split_col.to_string());
                }
            }
        }
        if self.permute_core_columns_sk && all_primary_columns.len() > 2 {
            let tail_len = all_primary_columns.len() - 1;
            for i in 0..tail_len {
                let j = ((self.seed as usize).wrapping_add(i * 7 + 3)) % tail_len;
                all_primary_columns.swap(1 + i, 1 + j);
            }
        }

        let create_sql = build_primary_create_table_ddl(
            &self.primary_table,
            &all_primary_columns,
            is_uuid_slug_pk,
            composite_tenant_code.is_some(),
            has_arch_m2,
            has_ambiguous_cand,
            is_numeric_tier_iso,
            underflow_floor,
        );
        let mut setup_sql = vec![create_sql];

        let needle_guard_vals = policy_matrix
            .synthesize_witness_row(self.needle_policy_target)
            .unwrap_or_else(|| vec![0; policy_arity]);
        let allow_guard_vals = policy_matrix
            .synthesize_witness_row(PolicyState::Allow)
            .unwrap_or_else(|| vec![1; policy_arity]);
        let deny_guard_vals = policy_matrix
            .synthesize_witness_row(PolicyState::Deny)
            .unwrap_or_else(|| vec![0; policy_arity]);

        let non_needle_slugs: Vec<String> = swarm
            .identifiers
            .iter()
            .enumerate()
            .filter(|(idx, _)| *idx != swarm.target_index)
            .map(|(_, s)| s.clone())
            .collect();

        let formatter = PrimaryTableRowFormatter {
            spec: self,
            all_primary_columns: &all_primary_columns,
            deceptive_decoy_columns: &deceptive_decoy_columns,
            homonym_lookalike_columns: &homonym_lookalike_columns,
            composite_tenant_code: composite_tenant_code.as_deref(),
            is_numeric_tier_iso,
        };

        let mut swarm_cursor = 0usize;
        let mut allow_clear_decoy_count = 0usize;
        let non_needle_total = total_rows.saturating_sub(1);
        let max_non_allow_decoys = non_needle_total / 20;
        let mut non_allow_decoys_emitted = 0usize;

        for chunk_start in (0..total_rows).step_by(200) {
            let chunk_end = (chunk_start + 200).min(total_rows);
            let mut row_tuples = Vec::with_capacity(chunk_end - chunk_start);
            for row_i in chunk_start..chunk_end {
                let pk = self.primary_key_iso.resolve_row_pk(row_i);
                if row_i == needle_idx {
                    row_tuples.push(formatter.format_tuple(
                        pk,
                        &needle_slug,
                        self.needle_initial_balance,
                        self.needle_policy_target,
                        self.needle_lock_target,
                        "steward_ops",
                        "ACTIVE",
                        &needle_guard_vals,
                    ));
                } else {
                    let s = if swarm_cursor < non_needle_slugs.len() {
                        let v = non_needle_slugs[swarm_cursor].clone();
                        swarm_cursor += 1;
                        v
                    } else {
                        format!("bystander_node_{:04}", row_i)
                    };
                    let (pol_st, lck_st, gv) =
                        if non_allow_decoys_emitted < max_non_allow_decoys && row_i % 23 == 19 {
                            non_allow_decoys_emitted += 1;
                            (PolicyState::Deny, LockState::Locked, &deny_guard_vals)
                        } else {
                            allow_clear_decoy_count += 1;
                            (PolicyState::Allow, LockState::Clear, &allow_guard_vals)
                        };
                    let decoy_bal = 100 + ((row_i as i64 * 37) % 900);
                    row_tuples.push(formatter.format_tuple(
                        pk,
                        &s,
                        decoy_bal,
                        pol_st,
                        lck_st,
                        "steward_auto",
                        "ACTIVE",
                        gv,
                    ));
                }
            }
            setup_sql.push(format!(
                "INSERT INTO {} VALUES {};",
                self.primary_table,
                row_tuples.join(", ")
            ));
        }

        let secondary = append_trap_rows_and_secondary_tables(
            self,
            &formatter,
            &mut setup_sql,
            needle_pk,
            &needle_slug,
            &needle_guard_vals,
            &allow_guard_vals,
            has_arch_m2,
            has_ambiguous_cand,
        );

        UnifiedSqlScenarioInstance {
            spec: self.clone(),
            highest_band: self.stress_vector.highest_band(),
            needle_row_index: needle_idx,
            needle_primary_key: needle_pk,
            needle_surface_key,
            needle_entity_slug: needle_slug,
            archived_honeypot_pk: secondary.archived_honeypot_pk,
            duplicate_candidate_pk: secondary.duplicate_candidate_pk,
            topic_switch_secondary_table: secondary.topic_switch_secondary_table,
            web_homonym_target_table: secondary.web_homonym_target_table,
            identifier_swarm: swarm,
            policy_matrix,
            canonical_columns,
            policy_guard_columns,
            deceptive_decoy_columns,
            homonym_lookalike_columns,
            noise_telemetry_columns,
            all_primary_columns,
            fk_chain_tables: secondary.fk_chain_tables,
            distractor_tables: secondary.distractor_tables,
            allow_clear_decoy_count,
            setup_sql,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_primary_create_table_ddl(
    primary_table: &str,
    all_primary_columns: &[String],
    is_uuid_slug_pk: bool,
    has_composite_tenant_pk: bool,
    has_arch_m2: bool,
    has_ambiguous_cand: bool,
    is_numeric_tier_iso: bool,
    underflow_floor: Option<i64>,
) -> String {
    let mut col_defs: Vec<String> = all_primary_columns
        .iter()
        .map(|c| {
            if c == "record_id" {
                if is_uuid_slug_pk || has_composite_tenant_pk {
                    "record_id INTEGER NOT NULL UNIQUE".to_string()
                } else {
                    "record_id INTEGER PRIMARY KEY".to_string()
                }
            } else if c == "entity_slug" {
                if is_uuid_slug_pk && !has_arch_m2 && !has_ambiguous_cand {
                    "entity_slug TEXT PRIMARY KEY".to_string()
                } else {
                    "entity_slug TEXT NOT NULL".to_string()
                }
            } else if c == "tenant_code" {
                "tenant_code TEXT NOT NULL".to_string()
            } else if c == "balance_credits" {
                if let Some(floor) = underflow_floor {
                    format!("balance_credits INTEGER NOT NULL CHECK (balance_credits >= {floor})")
                } else {
                    "balance_credits INTEGER NOT NULL".to_string()
                }
            } else if ((c == "policy_decision" || c == "lock_state") && is_numeric_tier_iso)
                || c.starts_with("policy_guard_c")
                || c.ends_with("_bit")
                || c.contains("balance")
                || c.contains("quota")
                || c.contains("timer")
            {
                format!("{c} INTEGER NOT NULL")
            } else {
                format!("{c} TEXT NOT NULL")
            }
        })
        .collect();
    if has_composite_tenant_pk {
        col_defs.push("PRIMARY KEY (tenant_code, record_id)".to_string());
    } else if is_uuid_slug_pk && (has_arch_m2 || has_ambiguous_cand) {
        col_defs.push("PRIMARY KEY (entity_slug, record_id)".to_string());
    }
    format!("CREATE TABLE {} ({});", primary_table, col_defs.join(", "))
}
