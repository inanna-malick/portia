//! Result inspection helpers: CSV (RFC-4180) & JSON `/export` and `/rows <N|all>` parsing.

use crate::DriverObservation;
use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use std::io::Write;
use std::path::Path;

/// Target file format for `/export <path.csv|path.json>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Csv,
    Json,
}

impl ExportFormat {
    pub fn from_path(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("csv") => Ok(Self::Csv),
            Some("json") => Ok(Self::Json),
            _ => bail!(
                "Unsupported export extension on {:?}; pass a .csv or .json file path (e.g. /export results.csv or /export results.json)",
                path
            ),
        }
    }

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Csv => "CSV",
            Self::Json => "JSON",
        }
    }
}

/// Parse the argument to `/rows <N|all>` into `Option<usize>` (`None` = unlimited).
pub fn parse_row_display_limit_arg(raw: &str) -> Result<Option<usize>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("Usage: /rows <N|all> (e.g. /rows 25 or /rows all)");
    }
    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "all" | "unlimited" | "none" | "inf" | "*"
    ) {
        return Ok(None);
    }
    let n: usize = trimmed.parse().with_context(|| {
        format!("Invalid row limit {trimmed:?}; expected positive integer or 'all'")
    })?;
    if n == 0 {
        bail!("Row display limit must be >= 1 or 'all'");
    }
    Ok(Some(n))
}

fn format_csv_cell(val: &Value) -> String {
    let raw = match val {
        Value::Null => return String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if raw.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", raw.replace('"', "\"\""))
    } else {
        raw
    }
}

fn resolved_column_names(obs: &DriverObservation) -> Vec<String> {
    if !obs.columns.is_empty() {
        obs.columns.clone()
    } else {
        let width = obs.rows.first().map_or(0, Vec::len);
        (1..=width).map(|i| format!("col_{i}")).collect()
    }
}

/// Serialize `obs` as RFC-4180 CSV bytes (header row + data rows).
#[must_use]
pub fn format_observation_csv(obs: &DriverObservation) -> String {
    let cols = resolved_column_names(obs);
    if cols.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    let header_cells: Vec<String> = cols
        .iter()
        .map(|c| format_csv_cell(&Value::String(c.clone())))
        .collect();
    out.push_str(&header_cells.join(","));
    out.push('\n');

    for row in &obs.rows {
        let line_cells: Vec<String> = (0..cols.len())
            .map(|i| format_csv_cell(row.get(i).unwrap_or(&Value::Null)))
            .collect();
        out.push_str(&line_cells.join(","));
        out.push('\n');
    }
    out
}

/// Serialize `obs` as a pretty JSON array of `{column: value}` row objects.
pub fn format_observation_json(obs: &DriverObservation) -> Result<String> {
    let cols = resolved_column_names(obs);
    let mut objects = Vec::with_capacity(obs.rows.len());
    for row in &obs.rows {
        let mut map = Map::with_capacity(cols.len());
        for (idx, col) in cols.iter().enumerate() {
            let val = row.get(idx).cloned().unwrap_or(Value::Null);
            map.insert(col.clone(), val);
        }
        objects.push(Value::Object(map));
    }
    Ok(serde_json::to_string_pretty(&Value::Array(objects))?)
}

/// Export `obs` to `path` (`.csv` or `.json`), creating parent directories if needed.
/// Returns `(format, row_count, column_count)`.
pub fn export_observation_to_path(
    obs: &DriverObservation,
    path: &Path,
) -> Result<(ExportFormat, usize, usize)> {
    let format = ExportFormat::from_path(path)?;
    let cols = resolved_column_names(obs);
    if cols.is_empty() && obs.rows.is_empty() {
        bail!("Latest observation contains 0 tabular columns/rows to export");
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create parent directory {:?}", parent))?;
        }
    }
    let payload = match format {
        ExportFormat::Csv => format_observation_csv(obs),
        ExportFormat::Json => format!("{}\n", format_observation_json(obs)?),
    };
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("failed to open export target {:?}", path))?;
    file.write_all(payload.as_bytes())
        .with_context(|| format!("failed writing export payload to {:?}", path))?;
    file.flush()?;
    Ok((format, obs.rows.len(), cols.len()))
}
