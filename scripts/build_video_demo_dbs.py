#!/usr/bin/env python3
"""Build realistic, non-trivial SQLite databases for the Portia TUI demo video.

Generates two rich, zero-decoy relational databases in `examples/`:
  1. `examples/cloud_ops_production.sqlite` (10 tables, 54 rows, 4 foreign-key relationships):
     A recognizable Cloud Backbone, Peering, Datacenter Facility, & Customer SLA Ledger database
     whose table/column names align 100% with the pre-calibrated `portia_v3_unified_cartridge.json`
     (`bgp_incidents`, `sla_credit_ledgers`, `pop_facility_sites`, `pop_facility_cooling_loops`,
     `oncall_sre_roster`, `bgp_peer_sessions`, `bgp_session_flap_links`, `mitigation_playbooks`,
     `maintenance_windows`, `bgp_incidents_audit_log`) with zero `_deprecated` decoy columns.
  2. `examples/saas_workspace_billing.sqlite` (8 tables, 52 rows, 5 foreign-key relationships):
     A classic B2B SaaS / Stripe-style Workspace, Subscription, Invoice, SLA Credit, & Support
     Escalation database (`workspace_accounts`, `subscription_plans`, `customer_invoices`,
     `sla_credit_notes`, `support_escalations`, `api_rate_policies`, `webhook_deliveries`,
     `billing_audit_events`).
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


def build_cloud_ops_production_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.executescript(
        """
        CREATE TABLE bgp_incidents (
            incident_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL,
            lifecycle_state TEXT NOT NULL,
            drop_rate_bps INTEGER NOT NULL,
            mitigation_rule TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            oncall_sre TEXT NOT NULL,
            change_freeze TEXT NOT NULL CHECK (change_freeze IN ('CLEAR', 'LOCKED')),
            FOREIGN KEY (pop_code) REFERENCES pop_facility_sites(pop_code) ON DELETE RESTRICT
        );

        CREATE TABLE pop_facility_sites (
            site_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL UNIQUE,
            metro_region TEXT NOT NULL,
            power_load_kw INTEGER NOT NULL,
            facility_steward TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            lock_state TEXT NOT NULL CHECK (lock_state IN ('CLEAR', 'LOCKED'))
        );

        CREATE TABLE pop_facility_cooling_loops (
            loop_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL,
            coolant_temp_c INTEGER NOT NULL,
            pump_rpm INTEGER NOT NULL,
            alarm_state TEXT NOT NULL,
            FOREIGN KEY (pop_code) REFERENCES pop_facility_sites(pop_code) ON DELETE RESTRICT
        );

        CREATE TABLE oncall_sre_roster (
            sre_id INTEGER PRIMARY KEY,
            sre_handle TEXT NOT NULL UNIQUE,
            pager_tier TEXT NOT NULL,
            home_pop_code TEXT NOT NULL,
            active_incidents_count INTEGER NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            FOREIGN KEY (home_pop_code) REFERENCES pop_facility_sites(pop_code) ON DELETE RESTRICT
        );

        CREATE TABLE mitigation_playbooks (
            playbook_id INTEGER PRIMARY KEY,
            rule_code TEXT NOT NULL UNIQUE,
            action_kind TEXT NOT NULL,
            auto_rollback_sec INTEGER NOT NULL,
            approver_role TEXT NOT NULL
        );

        CREATE TABLE bgp_session_flap_links (
            flap_id INTEGER PRIMARY KEY,
            incident_id INTEGER NOT NULL,
            peer_asn TEXT NOT NULL,
            flap_count_1h INTEGER NOT NULL,
            FOREIGN KEY (incident_id) REFERENCES bgp_incidents(incident_id) ON DELETE CASCADE
        );

        CREATE TABLE bgp_peer_sessions (
            session_id INTEGER PRIMARY KEY,
            peer_asn TEXT NOT NULL,
            pop_code TEXT NOT NULL,
            session_state TEXT NOT NULL,
            max_prefixes INTEGER NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            change_freeze TEXT NOT NULL CHECK (change_freeze IN ('CLEAR', 'LOCKED')),
            FOREIGN KEY (pop_code) REFERENCES pop_facility_sites(pop_code) ON DELETE RESTRICT
        );

        CREATE TABLE bgp_peer_sessions_history (
            history_id INTEGER PRIMARY KEY,
            session_id INTEGER NOT NULL,
            peer_asn TEXT NOT NULL,
            downtime_sec INTEGER NOT NULL,
            FOREIGN KEY (session_id) REFERENCES bgp_peer_sessions(session_id) ON DELETE CASCADE
        );

        CREATE TABLE sla_credit_ledgers (
            ledger_id INTEGER PRIMARY KEY,
            customer_slug TEXT NOT NULL,
            incident_id INTEGER NOT NULL,
            credit_usd INTEGER NOT NULL,
            payout_status TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            lock_state TEXT NOT NULL CHECK (lock_state IN ('CLEAR', 'LOCKED')),
            FOREIGN KEY (incident_id) REFERENCES bgp_incidents(incident_id) ON DELETE RESTRICT
        );

        CREATE TABLE maintenance_windows (
            window_id INTEGER PRIMARY KEY,
            pop_code TEXT NOT NULL,
            window_status TEXT NOT NULL,
            lead_sre TEXT NOT NULL,
            duration_min INTEGER NOT NULL,
            FOREIGN KEY (pop_code) REFERENCES pop_facility_sites(pop_code) ON DELETE RESTRICT
        );

        CREATE TABLE bgp_incidents_audit_log (
            audit_id INTEGER PRIMARY KEY,
            incident_id INTEGER NOT NULL,
            actor_handle TEXT NOT NULL,
            prev_drop_rate_bps INTEGER NOT NULL,
            new_drop_rate_bps INTEGER NOT NULL,
            FOREIGN KEY (incident_id) REFERENCES bgp_incidents(incident_id) ON DELETE CASCADE
        );
        """
    )

    conn.executemany(
        "INSERT INTO pop_facility_sites VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "NRT-TOKYO-04", "APAC-JP-EAST", 4200, "sre-japan-lead", "ALLOW", "CLEAR"),
            (2, "FRA-DECIX-02", "EU-CENTRAL-DE", 5100, "sre-eu-central", "ASK", "CLEAR"),
            (3, "IAD-ASHBURN-01", "US-EAST-VA", 6800, "sre-us-east", "DENY", "LOCKED"),
            (4, "SIN-EQUINIX-03", "APAC-SG-SOUTH", 3900, "sre-apac-south", "ALLOW", "CLEAR"),
            (5, "LHR-LINX-05", "EU-WEST-UK", 4600, "sre-uk-west", "ASK", "CLEAR"),
            (6, "AMS-AMSIX-07", "EU-WEST-NL", 4850, "sre-nl-core", "ALLOW", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO pop_facility_cooling_loops VALUES (?, ?, ?, ?, ?)",
        [
            (21, "NRT-TOKYO-04", 18, 2400, "NOMINAL"),
            (22, "FRA-DECIX-02", 27, 3100, "ELEVATED_TEMP"),
            (23, "IAD-ASHBURN-01", 34, 3600, "CRITICAL_CHILLER"),
            (24, "SIN-EQUINIX-03", 21, 2650, "NOMINAL"),
            (25, "LHR-LINX-05", 24, 2900, "WARN_FLOW_DELTA"),
            (26, "AMS-AMSIX-07", 19, 2500, "NOMINAL"),
        ],
    )

    conn.executemany(
        "INSERT INTO oncall_sre_roster VALUES (?, ?, ?, ?, ?, ?)",
        [
            (31, "kenji_sato", "PRIMARY_P1", "NRT-TOKYO-04", 1, "ALLOW"),
            (32, "lukas_weber", "PRIMARY_P1", "FRA-DECIX-02", 2, "ASK"),
            (33, "maya_lin", "INCIDENT_COMMANDER", "IAD-ASHBURN-01", 3, "DENY"),
            (34, "priya_nair", "SECONDARY_P2", "SIN-EQUINIX-03", 1, "ALLOW"),
            (35, "oliver_grant", "PRIMARY_P1", "LHR-LINX-05", 1, "ASK"),
            (36, "elena_rostova", "SECONDARY_P2", "AMS-AMSIX-07", 0, "ALLOW"),
        ],
    )

    conn.executemany(
        "INSERT INTO mitigation_playbooks VALUES (?, ?, ?, ?, ?)",
        [
            (41, "DRAIN_PENDING", "TRAFFIC_SHIFT", 300, "ONCALL_PRIMARY"),
            (42, "MED_BUMP_100", "BGP_MED_ADJUST", 600, "STEWARD_APPROVAL"),
            (43, "FREEZE_LOCKED", "HARD_LOCK", 0, "VP_INFRA"),
            (44, "LOCAL_PREF_80", "BGP_PREF_LOWER", 900, "ONCALL_PRIMARY"),
            (45, "FLOWSPEC_DROP", "DDOS_SCRUB_FILTER", 180, "STEWARD_APPROVAL"),
        ],
    )

    conn.executemany(
        "INSERT INTO bgp_incidents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (101, "NRT-TOKYO-04", "MITIGATING", 180, "DRAIN_PENDING", "ALLOW", "kenji_sato", "CLEAR"),
            (102, "FRA-DECIX-02", "INVESTIGATING", 940, "MED_BUMP_100", "ASK", "lukas_weber", "CLEAR"),
            (103, "IAD-ASHBURN-01", "ESCALATED", 3200, "FREEZE_LOCKED", "DENY", "maya_lin", "LOCKED"),
            (104, "SIN-EQUINIX-03", "MONITORING", 110, "LOCAL_PREF_80", "ALLOW", "priya_nair", "CLEAR"),
            (105, "LHR-LINX-05", "INVESTIGATING", 620, "FLOWSPEC_DROP", "ASK", "oliver_grant", "CLEAR"),
            (106, "AMS-AMSIX-07", "MITIGATING", 240, "DRAIN_PENDING", "ALLOW", "elena_rostova", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO bgp_session_flap_links VALUES (?, ?, ?, ?)",
        [
            (1, 101, "AS2914-NTT", 14),
            (2, 102, "AS3320-DTAG", 39),
            (3, 102, "AS1299-ARELION", 22),
            (4, 103, "AS13335-CLOUDFLARE", 84),
            (5, 105, "AS54113-FASTLY", 19),
            (6, 106, "AS174-COGENT", 11),
        ],
    )

    conn.executemany(
        "INSERT INTO bgp_peer_sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (11, "AS2914-NTT", "NRT-TOKYO-04", "ESTABLISHED", 250000, "ALLOW", "CLEAR"),
            (12, "AS3320-DTAG", "FRA-DECIX-02", "FLAPPING", 180000, "ASK", "CLEAR"),
            (13, "AS7018-ATT", "IAD-ASHBURN-01", "DAMPENED", 500000, "DENY", "LOCKED"),
            (14, "AS4637-TELSTRA", "SIN-EQUINIX-03", "ESTABLISHED", 150000, "ALLOW", "CLEAR"),
            (15, "AS54113-FASTLY", "LHR-LINX-05", "FLAPPING", 220000, "ASK", "CLEAR"),
            (16, "AS174-COGENT", "AMS-AMSIX-07", "ESTABLISHED", 300000, "ALLOW", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO bgp_peer_sessions_history VALUES (?, ?, ?, ?)",
        [
            (501, 11, "AS2914-NTT", 45),
            (502, 12, "AS3320-DTAG", 820),
            (503, 13, "AS7018-ATT", 2410),
            (504, 15, "AS54113-FASTLY", 390),
        ],
    )

    conn.executemany(
        "INSERT INTO sla_credit_ledgers VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (51, "discord-voice-apac", 101, 1200, "PENDING_CALC", "ALLOW", "CLEAR"),
            (52, "stripe-payments-eu", 102, 8500, "STEWARD_REVIEW", "ASK", "CLEAR"),
            (53, "linear-sync-us-east", 103, 45000, "FROZEN_AUDIT", "DENY", "LOCKED"),
            (54, "shopify-edge-sg", 104, 650, "PENDING_CALC", "ALLOW", "CLEAR"),
            (55, "figma-realtime-uk", 105, 4200, "STEWARD_REVIEW", "ASK", "CLEAR"),
            (56, "vercel-anycast-nl", 106, 1800, "APPROVED_CREDIT", "ALLOW", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO maintenance_windows VALUES (?, ?, ?, ?, ?)",
        [
            (61, "NRT-TOKYO-04", "COMPLETED", "kenji_sato", 45),
            (62, "FRA-DECIX-02", "IN_PROGRESS", "lukas_weber", 120),
            (63, "IAD-ASHBURN-01", "EMERGENCY_HOLD", "maya_lin", 240),
            (64, "LHR-LINX-05", "SCHEDULED", "oliver_grant", 90),
            (65, "AMS-AMSIX-07", "COMPLETED", "elena_rostova", 60),
        ],
    )

    conn.executemany(
        "INSERT INTO bgp_incidents_audit_log VALUES (?, ?, ?, ?, ?)",
        [
            (301, 101, "kenji_sato", 420, 180),
            (302, 102, "lukas_weber", 1100, 940),
            (303, 105, "oliver_grant", 890, 620),
            (304, 106, "elena_rostova", 510, 240),
        ],
    )

    conn.commit()
    conn.close()


def build_saas_workspace_billing_db(db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")

    conn.executescript(
        """
        CREATE TABLE workspace_accounts (
            workspace_id INTEGER PRIMARY KEY,
            org_slug TEXT NOT NULL UNIQUE,
            plan_tier TEXT NOT NULL,
            monthly_spend_usd INTEGER NOT NULL,
            seat_count INTEGER NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            account_owner TEXT NOT NULL,
            change_freeze TEXT NOT NULL CHECK (change_freeze IN ('CLEAR', 'LOCKED'))
        );

        CREATE TABLE subscription_plans (
            plan_id INTEGER PRIMARY KEY,
            plan_code TEXT NOT NULL UNIQUE,
            base_price_usd INTEGER NOT NULL,
            included_seats INTEGER NOT NULL,
            overage_rate_usd INTEGER NOT NULL,
            sla_uptime_bps INTEGER NOT NULL
        );

        CREATE TABLE customer_invoices (
            invoice_id INTEGER PRIMARY KEY,
            workspace_id INTEGER NOT NULL,
            billing_month TEXT NOT NULL,
            amount_due_usd INTEGER NOT NULL,
            invoice_status TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            lock_state TEXT NOT NULL CHECK (lock_state IN ('CLEAR', 'LOCKED')),
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE RESTRICT
        );

        CREATE TABLE sla_credit_notes (
            credit_note_id INTEGER PRIMARY KEY,
            invoice_id INTEGER NOT NULL,
            workspace_id INTEGER NOT NULL,
            credit_amount_usd INTEGER NOT NULL,
            reason_code TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            lock_state TEXT NOT NULL CHECK (lock_state IN ('CLEAR', 'LOCKED')),
            FOREIGN KEY (invoice_id) REFERENCES customer_invoices(invoice_id) ON DELETE RESTRICT,
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE RESTRICT
        );

        CREATE TABLE support_escalations (
            ticket_id INTEGER PRIMARY KEY,
            workspace_id INTEGER NOT NULL,
            severity_level TEXT NOT NULL,
            summary_slug TEXT NOT NULL,
            assigned_engineer TEXT NOT NULL,
            ticket_state TEXT NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE CASCADE
        );

        CREATE TABLE api_rate_policies (
            rate_policy_id INTEGER PRIMARY KEY,
            workspace_id INTEGER NOT NULL,
            burst_rps_limit INTEGER NOT NULL,
            daily_token_quota_m INTEGER NOT NULL,
            throttle_mode TEXT NOT NULL,
            policy_decision TEXT NOT NULL CHECK (policy_decision IN ('ALLOW', 'ASK', 'DENY')),
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE CASCADE
        );

        CREATE TABLE webhook_deliveries (
            delivery_id INTEGER PRIMARY KEY,
            workspace_id INTEGER NOT NULL,
            endpoint_host TEXT NOT NULL,
            http_status INTEGER NOT NULL,
            retry_count INTEGER NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE CASCADE
        );

        CREATE TABLE billing_audit_events (
            event_id INTEGER PRIMARY KEY,
            workspace_id INTEGER NOT NULL,
            actor_email TEXT NOT NULL,
            action_name TEXT NOT NULL,
            delta_usd INTEGER NOT NULL,
            FOREIGN KEY (workspace_id) REFERENCES workspace_accounts(workspace_id) ON DELETE CASCADE
        );
        """
    )

    conn.executemany(
        "INSERT INTO workspace_accounts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1001, "raycast-team-eu", "ENTERPRISE", 4800, 64, "ALLOW", "claire_dubois", "CLEAR"),
            (1002, "zed-editor-cloud", "ENTERPRISE_PLUS", 12400, 140, "ASK", "nathan_sobo", "CLEAR"),
            (1003, "oxide-rack-fleet", "MISSION_CRITICAL", 28500, 210, "DENY", "bryan_cantrill", "LOCKED"),
            (1004, "val-town-runtime", "GROWTH_PRO", 1950, 28, "ALLOW", "steve_krouse", "CLEAR"),
            (1005, "turso-edge-db", "ENTERPRISE", 7200, 85, "ASK", "glauber_costa", "CLEAR"),
            (1006, "fly-machines-ops", "ENTERPRISE_PLUS", 16800, 175, "ALLOW", "kurt_mackey", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO subscription_plans VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "GROWTH_PRO", 1500, 25, 45, 9990),
            (2, "ENTERPRISE", 4500, 60, 65, 9995),
            (3, "ENTERPRISE_PLUS", 10000, 125, 80, 9999),
            (4, "MISSION_CRITICAL", 25000, 200, 110, 10000),
        ],
    )

    conn.executemany(
        "INSERT INTO customer_invoices VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (2001, 1001, "2026-11", 4800, "OPEN", "ALLOW", "CLEAR"),
            (2002, 1002, "2026-11", 12400, "DISPUTED_OVERAGE", "ASK", "CLEAR"),
            (2003, 1003, "2026-11", 28500, "SETTLED_LOCKED", "DENY", "LOCKED"),
            (2004, 1004, "2026-11", 1950, "OPEN", "ALLOW", "CLEAR"),
            (2005, 1005, "2026-11", 7200, "PENDING_APPROVAL", "ASK", "CLEAR"),
            (2006, 1006, "2026-11", 16800, "OPEN", "ALLOW", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO sla_credit_notes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (3001, 2001, 1001, 450, "EU_WEST_LATENCY_SPIKE", "ALLOW", "CLEAR"),
            (3002, 2002, 1002, 1800, "COLLAB_RELAY_DROP", "ASK", "CLEAR"),
            (3003, 2003, 1003, 5000, "ANNUAL_AUDIT_FREEZE", "DENY", "LOCKED"),
            (3004, 2005, 1005, 950, "REPLICA_SYNC_DELAY", "ASK", "CLEAR"),
            (3005, 2006, 1006, 1200, "ANYCAST_FLAP_CREDIT", "ALLOW", "CLEAR"),
        ],
    )

    conn.executemany(
        "INSERT INTO support_escalations VALUES (?, ?, ?, ?, ?, ?)",
        [
            (4001, 1001, "SEV2", "sso_saml_token_skew", "aria_chen", "MITIGATING"),
            (4002, 1002, "SEV1", "crdt_buffer_rate_limit", "devon_hart", "AWAITING_STEWARD"),
            (4003, 1003, "SEV1", "hardware_hsm_seal_lock", "mira_patel", "ESCALATED_LOCKED"),
            (4004, 1004, "SEV3", "cron_worker_cold_start", "aria_chen", "RESOLVED"),
            (4005, 1005, "SEV2", "libsql_wal_checkpoint", "devon_hart", "INVESTIGATING"),
            (4006, 1006, "SEV2", "wireguard_mesh_mtu", "mira_patel", "MITIGATING"),
        ],
    )

    conn.executemany(
        "INSERT INTO api_rate_policies VALUES (?, ?, ?, ?, ?, ?)",
        [
            (5001, 1001, 1200, 50, "TOKEN_BUCKET", "ALLOW"),
            (5002, 1002, 3500, 180, "ADAPTIVE_BURST", "ASK"),
            (5003, 1003, 8000, 500, "DEDICATED_PIPE", "DENY"),
            (5004, 1004, 600, 20, "TOKEN_BUCKET", "ALLOW"),
            (5005, 1005, 2400, 110, "ADAPTIVE_BURST", "ASK"),
            (5006, 1006, 5000, 300, "DEDICATED_PIPE", "ALLOW"),
        ],
    )

    conn.executemany(
        "INSERT INTO webhook_deliveries VALUES (?, ?, ?, ?, ?)",
        [
            (6001, 1001, "hooks.raycast.com", 200, 0),
            (6002, 1002, "billing.zed.dev", 503, 4),
            (6003, 1003, "ops.oxide.computer", 200, 0),
            (6004, 1005, "webhooks.turso.tech", 429, 2),
            (6005, 1006, "events.fly.io", 200, 0),
        ],
    )

    conn.executemany(
        "INSERT INTO billing_audit_events VALUES (?, ?, ?, ?, ?)",
        [
            (7001, 1001, "claire@raycast.com", "APPLY_SLA_CREDIT", -450),
            (7002, 1002, "nathan@zed.dev", "REQUEST_OVERAGE_WAIVER", -1800),
            (7003, 1004, "steve@val.town", "SEAT_UPGRADE_3_TO_28", 450),
            (7004, 1006, "kurt@fly.io", "APPLY_ANYCAST_CREDIT", -1200),
        ],
    )

    conn.commit()
    conn.close()


def main() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    cloud_db = EXAMPLES_DIR / "cloud_ops_production.sqlite"
    saas_db = EXAMPLES_DIR / "saas_workspace_billing.sqlite"
    build_cloud_ops_production_db(cloud_db)
    build_saas_workspace_billing_db(saas_db)
    print(f"Built {cloud_db} ({cloud_db.stat().st_size} bytes)")
    print(f"Built {saas_db} ({saas_db.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
