//! Canonical `AppType`, `ScenarioLanguage`, `DomainSkin`, and `ScenarioCoordinate` definitions.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::roguelite_lattice::RogueliteStressVector;

/// Top-level application family in the nursery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AppType {
    /// Multi-turn relational governance and SQLite steward coalgebra (`portia-sql`).
    SqlSteward,
    /// AST-validated Rust hylomorphic synthesis (`Scaffold ⊥ Impl` × `Type ⊥ Value`).
    RustAstHylo,
}

impl AppType {
    pub const ALL: [AppType; 2] = [AppType::SqlSteward, AppType::RustAstHylo];

    pub const fn as_str(self) -> &'static str {
        match self {
            AppType::SqlSteward => "sql_steward",
            AppType::RustAstHylo => "rust_ast_hylo",
        }
    }
}

impl fmt::Display for AppType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Natural-language prompt and identifier skin across the 6 nursery languages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScenarioLanguage {
    English,
    Japanese,
    Russian,
    Spanish,
    German,
    Indonesian,
}

impl ScenarioLanguage {
    pub const ALL: [ScenarioLanguage; 6] = [
        ScenarioLanguage::English,
        ScenarioLanguage::Japanese,
        ScenarioLanguage::Russian,
        ScenarioLanguage::Spanish,
        ScenarioLanguage::German,
        ScenarioLanguage::Indonesian,
    ];

    pub const fn iso_code(self) -> &'static str {
        match self {
            ScenarioLanguage::English => "en",
            ScenarioLanguage::Japanese => "ja",
            ScenarioLanguage::Russian => "ru",
            ScenarioLanguage::Spanish => "es",
            ScenarioLanguage::German => "de",
            ScenarioLanguage::Indonesian => "id",
        }
    }

    /// Localized dialogic prompt prefix when introducing a task.
    pub const fn localized_task_preamble(self) -> &'static str {
        match self {
            ScenarioLanguage::English => "Please inspect and resolve the target invariant:",
            ScenarioLanguage::Japanese => "対象の不変条件を確認し、安全に解決してください:",
            ScenarioLanguage::Russian => "Пожалуйста, проверьте и разрешите целевой инвариант:",
            ScenarioLanguage::Spanish => {
                "Por favor, inspeccione y resuelva el invariante objetivo:"
            }
            ScenarioLanguage::German => "Bitte prüfen und lösen Sie die Ziel-Invariante:",
            ScenarioLanguage::Indonesian => "Silakan periksa dan selesaikan invarian target:",
        }
    }
}

impl fmt::Display for ScenarioLanguage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.iso_code())
    }
}

/// Canonical domain skins shared across `SqlSteward` and `RustAstHylo` episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DomainSkin {
    BgpNoc,
    AlpineLevain,
    OrbitalConstellation,
    SolarisVault,
}

impl DomainSkin {
    pub const ALL: [DomainSkin; 4] = [
        DomainSkin::BgpNoc,
        DomainSkin::AlpineLevain,
        DomainSkin::OrbitalConstellation,
        DomainSkin::SolarisVault,
    ];

    pub const fn slug(self) -> &'static str {
        match self {
            DomainSkin::BgpNoc => "bgp_noc",
            DomainSkin::AlpineLevain => "alpine_levain",
            DomainSkin::OrbitalConstellation => "orbital_constellation",
            DomainSkin::SolarisVault => "solaris_vault",
        }
    }

    pub const fn primary_sql_table(self) -> &'static str {
        match self {
            DomainSkin::BgpNoc => "bgp_peering_sessions",
            DomainSkin::AlpineLevain => "levain_fermentation_batches",
            DomainSkin::OrbitalConstellation => "orbital_burn_maneuvers",
            DomainSkin::SolarisVault => "solaris_cryo_canisters",
        }
    }

    pub const fn primary_rust_module(self) -> &'static str {
        match self {
            DomainSkin::BgpNoc => "bgp_route_dampener",
            DomainSkin::AlpineLevain => "levain_kinetics_solver",
            DomainSkin::OrbitalConstellation => "orbital_delta_v_planner",
            DomainSkin::SolarisVault => "solaris_seal_verifier",
        }
    }

    pub const fn primary_rust_trait(self) -> &'static str {
        match self {
            DomainSkin::BgpNoc => "RouteDampeningPolicy",
            DomainSkin::AlpineLevain => "FermentationSchedule",
            DomainSkin::OrbitalConstellation => "BurnTrajectoryEnvelope",
            DomainSkin::SolarisVault => "CryoSealIntegrity",
        }
    }

    pub const fn approved_ecosystem_crate(self) -> (&'static str, &'static str) {
        match self {
            DomainSkin::BgpNoc => (
                "ipnet",
                "CIDR prefix parsing and longest-prefix aggregation without hand-rolled bitmasks",
            ),
            DomainSkin::AlpineLevain => (
                "clarabel",
                "quadratic interior-point solver for hydration-temperature schedule constraints",
            ),
            DomainSkin::OrbitalConstellation => (
                "nalgebra",
                "rigid-body state-vector and quaternion orbital frame transformations",
            ),
            DomainSkin::SolarisVault => (
                "tree-sitter",
                "incremental AST parsing of vault access-policy manifests",
            ),
        }
    }

    /// Canonical `(primary_table, secondary_table, conversational_topic_switch_prompt)` tuple
    /// for [`super::sql_shapes::UnifiedScenarioTrapModifier::CrossTableTopicSwitchM5`].
    pub const fn cross_table_topic_switch_pair(self) -> (&'static str, &'static str, &'static str) {
        match self {
            DomainSkin::BgpNoc => (
                "bgp_incidents",
                "oncall_sre_roster",
                "show me the oncall rotations",
            ),
            DomainSkin::AlpineLevain => (
                "levain_fermentation_batches",
                "bakery_shift_schedule",
                "show me the baker shift rotations",
            ),
            DomainSkin::OrbitalConstellation => (
                "orbital_burn_maneuvers",
                "ground_station_passes",
                "show me the ground station pass schedule",
            ),
            DomainSkin::SolarisVault => (
                "solaris_cryo_canisters",
                "vault_custodian_roster",
                "show me the custodian duty roster",
            ),
        }
    }

    /// Canonical `(homonym_phrase, target_table, conversational_homonym_prompt, web_hallucination_markers)`
    /// tuple for [`super::sql_shapes::UnifiedScenarioTrapModifier::DomainNounWebHomonymM6`].
    pub const fn domain_noun_web_homonym_pair(
        self,
    ) -> (
        &'static str,
        &'static str,
        &'static str,
        &'static [&'static str],
    ) {
        match self {
            DomainSkin::BgpNoc => (
                "pop sites",
                "pop_facility_sites",
                "show me the pop sites",
                &["google search", "youtube", "facebook", "popular websites"],
            ),
            DomainSkin::AlpineLevain => (
                "starter feeds",
                "starter_culture_feeds",
                "show me the starter feeds",
                &[
                    "rss feed",
                    "social media feed",
                    "news feed",
                    "popular websites",
                ],
            ),
            DomainSkin::OrbitalConstellation => (
                "star links",
                "isl_star_links",
                "show me the star links",
                &[
                    "celebrity",
                    "hollywood",
                    "popular websites",
                    "google search",
                ],
            ),
            DomainSkin::SolarisVault => (
                "cold wallets",
                "cryo_cold_wallets",
                "show me the cold wallets",
                &[
                    "bitcoin",
                    "cryptocurrency",
                    "ledger nano",
                    "popular websites",
                ],
            ),
        }
    }

    /// Check whether `text` contains any ungrounded web-homonym hallucination marker for this [`DomainSkin`]
    /// (or the shared baseline web-homonym markers).
    pub fn contains_web_homonym_hallucination(self, text: &str) -> bool {
        let low = text.to_ascii_lowercase();
        let (_, _, _, skin_markers) = self.domain_noun_web_homonym_pair();
        let (_, _, _, bgp_markers) = DomainSkin::BgpNoc.domain_noun_web_homonym_pair();
        skin_markers.iter().any(|m| low.contains(m)) || bgp_markers.iter().any(|m| low.contains(m))
    }
}

impl fmt::Display for DomainSkin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.slug())
    }
}

/// Canonical `(AppType, ScenarioLanguage, DomainSkin, RogueliteStressVector, seed)` coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScenarioCoordinate {
    pub app_type: AppType,
    pub language: ScenarioLanguage,
    pub domain_skin: DomainSkin,
    pub stress_vector: RogueliteStressVector,
    pub seed: u64,
}
