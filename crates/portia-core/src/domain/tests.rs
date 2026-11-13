use super::*;

#[test]
fn domain_primitives_roundtrip_and_classify_decoy_columns() {
    for p in PolicyState::ALL {
        assert_eq!(PolicyState::parse_flexible(p.as_canonical_str()), Some(p));
    }
    assert_eq!(
        PolicyState::parse_flexible("GREEN_AUTO"),
        Some(PolicyState::Allow)
    );
    assert_eq!(PolicyState::parse_flexible("1"), Some(PolicyState::Ask));
    assert_eq!(
        PolicyState::parse_flexible("RED_VETO"),
        Some(PolicyState::Deny)
    );

    for l in LockState::ALL {
        assert_eq!(LockState::parse_flexible(l.as_canonical_str()), Some(l));
    }
    assert!(!LockState::Clear.is_locked());
    assert!(LockState::Locked.is_locked());

    for s in RowLifecycleStatus::ALL {
        assert_eq!(
            RowLifecycleStatus::parse_flexible(s.as_canonical_str()),
            Some(s)
        );
    }
    assert!(RowLifecycleStatus::Archived.is_decommissioned_decoy());
    assert!(RowLifecycleStatus::Obsolete.is_decommissioned_decoy());
    assert!(!RowLifecycleStatus::Active.is_decommissioned_decoy());

    for op in WriteOp::ALL {
        assert_eq!(WriteOp::parse_flexible(op.sql_verb()), Some(op));
    }

    let live_col = ColumnSchema::new_canonical("poster_altitude_m", "INTEGER", false, false);
    let decoy_col = ColumnSchema::new_canonical(
        "legacy_poster_altitude_m_deprecated",
        "INTEGER",
        false,
        true,
    );
    assert!(!live_col.nullable());
    assert!(decoy_col.nullable());
    assert_eq!(live_col.alias_class(), ColumnAliasClass::CanonicalLive);
    assert_eq!(
        decoy_col.alias_class(),
        ColumnAliasClass::QuarantinedLegacyDecoy
    );

    let schema = TableSchema::new("alpine_levain_campaigns", vec![live_col, decoy_col])
        .with_foreign_keys(vec![ForeignKeyEdge {
            child_table: "alpine_poster_links".to_string(),
            child_fk_column: "campaign_id".to_string(),
            parent_table: "alpine_levain_campaigns".to_string(),
            parent_pk_column: "campaign_id".to_string(),
            on_delete: ForeignKeyDeleteRule::Restrict,
        }]);
    assert_eq!(schema.live_columns().len(), 1);
    assert_eq!(schema.decoy_columns().len(), 1);
}
