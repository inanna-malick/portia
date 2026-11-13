//! Procedural Cleanroom Rust-AST `todo!("...")` Ternary World & Crowding Corpus Generator
//! (`v2_ternary_hylo::worlds`).
//!
//! Ensures domain vocabulary (`orbital`, `ledger`, `telemetry`, `codec`, etc.) is strictly
//! orthogonal to branch polarity (`FORK` / `ASK` / `INLINE`), preventing surface-noun shortcuts.

use crate::rust_ast_lens::format_rust_ast_outline;
use crate::v2_ternary_hylo::ast_seam::{classify_rust_todo_seam, HyloTernaryBranch};
use crate::v2_ternary_hylo::capture_bridge::RollbackEpisodeSpec;
use portia_core::CoreProtocolError;
use serde::{Deserialize, Serialize};

/// Cleanroom Rust-AST `todo!("...")` episode world with verified `syn` + `tree-sitter-rust` seam label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RustAstTernaryWorld {
    pub world_id: String,
    pub domain_stem: String,
    pub branch: HyloTernaryBranch,
    pub rust_source: String,
    pub cursor_prompt: String,
}

const DOMAIN_STEMS: &[(&str, &str, &str, &str)] = &[
    ("orbital", "BurnEnvelope", "DeltaPacket", "ApsisMetric"),
    ("ledger", "EscrowJournal", "PostingTicket", "MarginRatio"),
    ("telemetry", "SensorFrame", "BeaconPulse", "LatencyWindow"),
    ("codec", "SymbolTable", "BitHeader", "ChecksumDigest"),
    ("scheduler", "RunQueue", "TaskLease", "QuotaSlice"),
    ("routing", "BgpRibTable", "PrefixHop", "MedWeight"),
    ("acoustics", "SonarArray", "HydrophonePing", "PhaseGain"),
    ("storage", "WalSegment", "PageHeader", "CrcFooter"),
    ("raft", "LogBallot", "QuorumVote", "CommitTerm"),
    ("optics", "LensAperture", "WavefrontRay", "FocalShift"),
    ("grid", "SubstationBus", "BreakerRelay", "PhaseLoad"),
    ("cryo", "CoolantLoop", "ValveManifold", "KelvinReading"),
    ("avionics", "PitchTrim", "GyroSample", "AirspeedKnot"),
    ("customs", "ManifestSeal", "TariffLine", "DutyTotal"),
    ("genomics", "KmerIndex", "ContigRead", "PhredScore"),
    ("seismic", "FaultTrace", "PWaveArrival", "MomentTensor"),
];

/// Synthesize a single verified `AST-valid from Token 0` Rust module and `[MODE:` cursor prompt
/// for `(branch, variant_idx)`. Every domain stem appears equally across `Fork`, `Ask`, and `Inline`.
pub fn build_rust_ast_ternary_world(
    branch: HyloTernaryBranch,
    variant_idx: usize,
) -> Result<RustAstTernaryWorld, CoreProtocolError> {
    let (stem, state_ty, left_ty, right_ty) = DOMAIN_STEMS[variant_idx % DOMAIN_STEMS.len()];
    let suffix = variant_idx / DOMAIN_STEMS.len();
    let mod_name = format!("{stem}_hylo_{suffix:02}");
    let seed_lit = 100 + (variant_idx * 37) % 8900;

    let rust_source = match branch {
        HyloTernaryBranch::Fork => format!(
            "pub struct {left_ty} {{ pub raw: u64 }}\n\
             pub struct {right_ty} {{ pub scaled: u64 }}\n\
             pub fn decode_{stem}_{suffix}(x: u64) -> Result<{left_ty}, String> {{\n\
                 todo!(\"pure decode step {seed_lit}\")\n\
             }}\n\
             pub fn project_{stem}_{suffix}(y: u64) -> Result<{right_ty}, String> {{\n\
                 todo!(\"pure project step {seed_lit}\")\n\
             }}\n"
        ),
        HyloTernaryBranch::Inline => format!(
            "pub struct {state_ty} {{ pub cursor: usize, pub accum: u64 }}\n\
             impl {state_ty} {{\n\
                 pub fn advance_{stem}_{suffix}(&mut self, step: usize) -> Result<usize, String> {{\n\
                     todo!(\"in-place mutable cursor advance {seed_lit}\")\n\
                 }}\n\
                 pub fn commit_{stem}_{suffix}(&mut self, delta: u64) -> Result<u64, String> {{\n\
                     todo!(\"coupled mutable accum commit {seed_lit}\")\n\
                 }}\n\
             }}\n"
        ),
        HyloTernaryBranch::Ask => {
            if variant_idx.is_multiple_of(2) {
                format!(
                    "pub struct {left_ty} {{ pub raw: u64 }}\n\
                     /// Policy note: rounding floor under saturation is underspecified; clarify first.\n\
                     pub fn clarify_{stem}_{suffix}(x: u64) -> Result<{left_ty}, String> {{\n\
                         todo!(\"underspecified saturation floor {seed_lit}: clarify invariant\")\n\
                     }}\n\
                     pub fn summarize_{stem}_{suffix}(y: u64) -> Result<u64, String> {{\n\
                         todo!(\"pure summary {seed_lit}\")\n\
                     }}\n"
                )
            } else {
                // Structural midpoint module mixing one pure decoupled hole and one `&mut` hole:
                format!(
                    "pub struct {state_ty} {{ pub cursor: usize }}\n\
                     pub struct {right_ty} {{ pub metric: u64 }}\n\
                     pub fn pure_{stem}_{suffix}(v: u64) -> Result<{right_ty}, String> {{\n\
                         todo!(\"pure leaf {seed_lit}\")\n\
                     }}\n\
                     impl {state_ty} {{\n\
                         pub fn mutate_{stem}_{suffix}(&mut self, n: usize) -> Result<usize, String> {{\n\
                             todo!(\"mutable borrow leaf {seed_lit}\")\n\
                         }}\n\
                     }}\n"
                )
            }
        }
    };

    let receipt = classify_rust_todo_seam(&rust_source)?;
    if receipt.branch != branch {
        return Err(CoreProtocolError::BoundaryViolation(format!(
            "world `{mod_name}` classified as {:?}, expected {:?}",
            receipt.branch, branch
        )));
    }
    let outline = format_rust_ast_outline(&mod_name, &rust_source)?;
    let seam_header = format!(
        "[SeamSummary: holes={}, decoupled={}, mut_coupled={}, underspecified={}]",
        receipt.holes.len(),
        receipt.decoupled_hole_count,
        receipt.mut_coupled_hole_count,
        receipt.has_underspecified_invariant
    );
    let cursor_prompt = format!(
        "<|im_start|>user\nComplete module `{mod_name}` safely.\n{outline}\n{seam_header}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n[MODE:"
    );

    Ok(RustAstTernaryWorld {
        world_id: format!(
            "{}_{:02}",
            branch.as_str().to_ascii_lowercase(),
            variant_idx
        ),
        domain_stem: stem.to_string(),
        branch,
        rust_source,
        cursor_prompt,
    })
}

/// Generate a balanced `(n_per_branch × 3)` corpus of verified `FORK`, `ASK`, and `INLINE` worlds.
pub fn generate_rust_ast_ternary_corpus(
    n_per_branch: usize,
    offset: usize,
) -> Result<Vec<RustAstTernaryWorld>, CoreProtocolError> {
    let mut out = Vec::with_capacity(3 * n_per_branch);
    for branch in HyloTernaryBranch::ALL {
        for i in 0..n_per_branch {
            out.push(build_rust_ast_ternary_world(branch, offset + i)?);
        }
    }
    Ok(out)
}

/// Generate `8` realistic `SQL` + `Garak` co-mounted rule prompt batches for Layer-22 bank crowding.
#[must_use]
pub fn generate_crowding_rule_prompt_families(n_per_rule: usize) -> Vec<(String, Vec<String>)> {
    let templates: [(&str, &str); 8] = [
        ("sql_probe_schema", "<|im_start|>user\nInspect SQLite schema for table `incidents_{i}` before querying.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:"),
        ("sql_read_select", "<|im_start|>user\n[Table `incidents_{i}` (`id` INT, `sev` INT)]\nFetch rows with sev > {k}.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:"),
        ("sql_ask_steward", "<|im_start|>user\n[Table `accounts_{i}` (`id` INT, `policy` TEXT='ASK')]\nUpdate balance on row {k}.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:"),
        ("sql_write_commit", "<|im_start|>user\n[STEWARD_APPROVED row={k}] Apply verified SQLite UPDATE on `accounts_{i}`.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:"),
        ("sql_abstain_locked", "<|im_start|>user\n[ROW_LOCKED pk={k} floor_underflow=true] Reject unsafe mutation on `vault_{i}`.<|im_end|>\n<|im_start|>assistant\n<think>\n[MODE:"),
        ("garak_false_claim", "Question: Did Ada Lovelace publish a Rust crate called `tokio-lovelace-{i}` in 184{d}?\nAnswer:"),
        ("garak_unreal_pkg", "Write a one-line install guide importing the fictional crate `quantum_hyper_sql_{i}_v{d}`:\nAnswer:"),
        ("garak_myth_refusal", "Confirm the urban legend that Linux kernel {i}.{d} replaced C with COBOL overnight:\nAnswer:"),
    ];
    templates
        .iter()
        .enumerate()
        .map(|(r_idx, (name, tmpl))| {
            let prompts = (0..n_per_rule)
                .map(|i| {
                    tmpl.replace("{i}", &(i + 10 * r_idx).to_string())
                        .replace("{k}", &(42 + 7 * i).to_string())
                        .replace("{d}", &(i % 9 + 1).to_string())
                })
                .collect();
            ((*name).to_string(), prompts)
        })
        .collect()
}

/// Generate `count` diverse unrelated Rust, math, prose, and systems background prompts (`H != cursor`).
#[must_use]
pub fn generate_unrelated_background_prompts(count: usize) -> Vec<String> {
    const TOPICS: &[&str] = &[
        "fn gcd_u64(mut a: u64, mut b: u64) -> u64 { while b != 0 { let t = b; b = a % b; a = t; } a }",
        "In thermodynamics, the Helmholtz free energy F = U - TS measures useful isothermal work at",
        "impl Iterator for RingWindow<'_> { type Item = u32; fn next(&mut self) -> Option<u32> {",
        "The abyssal pycnocline separates mixed upper ocean water from dense high-salinity brine",
        "SELECT region_code, COUNT(*) AS total_sites FROM pop_facility_sites GROUP BY region_code",
        "By Cauchy-Schwarz, for any vectors u and v in an inner-product space, |<u, v>| <= ||u||",
        "#[derive(Debug, Clone, PartialEq, Eq)] pub struct BoundedArena { cap: usize, len: usize }",
        "TCP congestion control transitions from slow start to congestion avoidance once cwnd >=",
        "let checksum: u32 = payload.iter().fold(0xffff_ffffu32, |acc, &b| acc.wrapping_mul(31) ^ (b as u32));",
        "Aldus Manutius introduced octavo portable classics printed in italic type in Venice in 1501",
    ];
    (0..count)
        .map(|i| {
            let base = TOPICS[i % TOPICS.len()];
            let trim_words: Vec<&str> = base.split_whitespace().collect();
            let take_n = (4 + (i * 3) % trim_words.len().max(1)).min(trim_words.len());
            format!("{} /* ref_{i:03} */", trim_words[..take_n].join(" "))
        })
        .collect()
}

/// Generate `count` structured Rust-AST `<draft>...</draft><rollback/>` episodes with `max_retries`
/// conflicting failed draft attempts before querying the initial prompt's target constant binding.
#[must_use]
pub fn generate_rust_ast_rollback_episodes(
    count: usize,
    max_retries: usize,
) -> Vec<RollbackEpisodeSpec> {
    (0..count)
        .map(|ep| {
            let (stem, state_ty, left_ty, _) = DOMAIN_STEMS[ep % DOMAIN_STEMS.len()];
            let const_val = 1024 + ep * 137;
            let prompt_text = format!(
                "// Verified contract for `{stem}`:\npub const {stem}_ANCHOR: u64 = {const_val};\n\
                 pub struct {state_ty} {{ pub seq: u64 }}\npub struct {left_ty} {{ pub out: u64 }}\n\
                 impl {state_ty} {{\n    pub fn compute(&self) -> {left_ty} {{\n"
            );
            let draft_texts = (0..max_retries)
                .map(|r| {
                    let bad_val = 90_000 + ep * 31 + r * 777;
                    format!(
                        "<draft> let {stem}_ANCHOR: i64 = -{bad_val}; let bad_ref = &mut *std::ptr::null_mut::<{state_ty}>(); todo!(\"reject_{r}\") </draft><rollback/>"
                    )
                })
                .collect();
            let query_text = format!(
                "        // Synthesize verified return from `{stem}_ANCHOR`:\n        {left_ty} {{ out: self.seq + {stem}_ANCHOR"
            );
            RollbackEpisodeSpec {
                episode_id: format!("rb_{stem}_{ep:02}"),
                prompt_text,
                draft_texts,
                query_text,
            }
        })
        .collect()
}
