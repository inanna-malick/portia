//! HTTP `curl` and `codex exec` Chat-Completion Transports (`ChatCompletionTransport`).

use super::*;

// ============================================================================
// 1. Pure-Rust `curl` Transport (`CurlChatCompletionTransport`)
// ============================================================================

/// Zero-extra-crate OpenAI `/v1/chat/completions` HTTP client backed by `curl`, piping serialized
/// JSON request payloads over `stdin` (`--data-binary @-`) so prompt payloads never bloat `argv`.
#[derive(Clone)]
pub struct CurlChatCompletionTransport {
    base_url: String,
    api_key: String,
    model: String,
    timeout_secs: u64,
}

impl std::fmt::Debug for CurlChatCompletionTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CurlChatCompletionTransport")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("timeout_secs", &self.timeout_secs)
            .finish()
    }
}

impl CurlChatCompletionTransport {
    /// Construct a [`CurlChatCompletionTransport`] with explicit configuration, failing closed if
    /// `api_key` or `base_url` is blank.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
        timeout_secs: u64,
    ) -> Result<Self> {
        let base_url = base_url.into().trim().to_string();
        if base_url.is_empty() {
            bail!("OpenAI base_url cannot be empty");
        }
        let api_key = api_key.into().trim().to_string();
        if api_key.is_empty() {
            bail!("OpenAI API key cannot be empty; set $OPENAI_API_KEY before running CurlChatCompletionTransport");
        }
        let raw_model = model.into();
        let clean_model = raw_model.trim();
        let model = if clean_model.is_empty() {
            DEFAULT_OPENAI_SYNTH_MODEL.to_string()
        } else {
            clean_model.to_string()
        };
        Ok(Self {
            base_url,
            api_key,
            model,
            timeout_secs: timeout_secs.max(1),
        })
    }

    /// Construct a [`CurlChatCompletionTransport`] from `$OPENAI_API_KEY` (required),
    /// `$OPENAI_BASE_URL` (default [`DEFAULT_OPENAI_BASE_URL`]), and `$OPENAI_SYNTH_MODEL`
    /// (default [`DEFAULT_OPENAI_SYNTH_MODEL`]).
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing or empty $OPENAI_API_KEY required by CurlChatCompletionTransport"
                )
            })?;
        let base_url = std::env::var("OPENAI_BASE_URL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string());
        let model = std::env::var("OPENAI_SYNTH_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_OPENAI_SYNTH_MODEL.to_string());
        Self::new(base_url, api_key, model, DEFAULT_CURL_TIMEOUT_SECS)
    }

    /// Override the base URL (`https://api.openai.com/v1` or a local loopback gateway).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Result<Self> {
        let trimmed = base_url.into().trim().to_string();
        if trimmed.is_empty() {
            bail!("OpenAI base_url cannot be empty");
        }
        self.base_url = trimmed;
        Ok(self)
    }

    /// Override the default chat model used when `ChatCompletionRequest.model` is blank.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        let trimmed = model.into().trim().to_string();
        if !trimmed.is_empty() {
            self.model = trimmed;
        }
        self
    }

    /// Override the per-request `curl --max-time` timeout in seconds.
    pub fn with_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs.max(1);
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs
    }

    /// Resolve the full `/chat/completions` endpoint URL from `self.base_url`.
    pub fn chat_completions_endpoint(&self) -> String {
        let trimmed = self.base_url.trim_end_matches('/');
        if trimmed.ends_with("/chat/completions") {
            trimmed.to_string()
        } else {
            format!("{trimmed}/chat/completions")
        }
    }
}

pub(super) fn parse_openai_chat_completion_response(
    body_str: &str,
    http_status: u16,
) -> Result<String> {
    let trimmed = body_str.trim();
    if trimmed.is_empty() {
        bail!("OpenAI chat/completions returned an empty body (HTTP {http_status})");
    }

    let parsed: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(val) => val,
        Err(err) => {
            let preview: String = trimmed.chars().take(240).collect();
            bail!(
                "failed to parse OpenAI chat/completions JSON (HTTP {http_status}): {err}; body={preview:?}"
            );
        }
    };

    if let Some(err_val) = parsed.get("error").filter(|v| !v.is_null()) {
        let err_msg = err_val
            .get("message")
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| err_val.to_string());
        let err_type = err_val
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or("api_error");
        bail!("OpenAI API error (HTTP {http_status}, {err_type}): {err_msg}");
    }

    if !(200..300).contains(&http_status) {
        let preview: String = trimmed.chars().take(240).collect();
        bail!("OpenAI chat/completions HTTP {http_status} error: {preview}");
    }

    let message = parsed
        .get("choices")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|choice| choice.get("message"))
        .ok_or_else(|| {
            anyhow::anyhow!("OpenAI chat/completions JSON missing `choices[0].message`: {trimmed}")
        })?;

    let content_val = message.get("content").ok_or_else(|| {
        anyhow::anyhow!(
            "OpenAI chat/completions JSON missing `choices[0].message.content`: {trimmed}"
        )
    })?;

    let extracted = if let Some(s) = content_val.as_str() {
        s.trim().to_string()
    } else if let Some(parts) = content_val.as_array() {
        let joined: Vec<&str> = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect();
        joined.join("").trim().to_string()
    } else {
        String::new()
    };

    if extracted.is_empty() {
        bail!("OpenAI chat/completions returned empty `choices[0].message.content`");
    }
    Ok(extracted)
}

struct PrivateHeaderFile {
    path: std::path::PathBuf,
}

impl PrivateHeaderFile {
    fn write_bearer_headers(api_key: &str) -> Result<Self> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let seq = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!(".portia_curl_hdr_{pid}_{seq}"));
        let mut opts = std::fs::OpenOptions::new();
        opts.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let mut file = opts
            .open(&path)
            .context("failed to create private 0600 curl header file")?;
        let clean_key: String = api_key
            .chars()
            .filter(|&c| c != '\r' && c != '\n')
            .collect();
        writeln!(file, "Authorization: Bearer {clean_key}")?;
        writeln!(file, "Content-Type: application/json")?;
        file.flush()?;
        Ok(Self { path })
    }
}

impl Drop for PrivateHeaderFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl ChatCompletionTransport for CurlChatCompletionTransport {
    fn complete_chat(&mut self, req: &ChatCompletionRequest) -> Result<String> {
        let effective_model = if req.model.trim().is_empty() {
            self.model.clone()
        } else {
            req.model.trim().to_string()
        };
        let effective_temperature = req
            .temperature
            .filter(|_| model_supports_custom_temperature(&effective_model));
        let wire_req = ChatCompletionRequest {
            model: effective_model,
            messages: req.messages.clone(),
            temperature: effective_temperature,
            max_completion_tokens: req.max_completion_tokens,
        };
        let payload_bytes = serde_json::to_vec(&wire_req)
            .context("failed to serialize ChatCompletionRequest to JSON")?;

        let endpoint = self.chat_completions_endpoint();
        let timeout_arg = self.timeout_secs.max(1).to_string();
        let hdr_guard = PrivateHeaderFile::write_bearer_headers(&self.api_key)?;
        let hdr_arg = format!("@{}", hdr_guard.path.display());

        let mut child = Command::new("curl")
            .arg("-sS")
            .arg("--max-time")
            .arg(&timeout_arg)
            .arg("-X")
            .arg("POST")
            .arg(&endpoint)
            .arg("-H")
            .arg(&hdr_arg)
            .arg("-w")
            .arg("\n%{http_code}")
            .arg("--data-binary")
            .arg("@-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("failed to spawn `curl` for OpenAI chat/completions transport")?;

        {
            let mut stdin = child
                .stdin
                .take()
                .context("failed to open stdin pipe to `curl` subprocess")?;
            stdin
                .write_all(&payload_bytes)
                .context("failed to write JSON request body to `curl` stdin")?;
        }

        let output = child
            .wait_with_output()
            .context("failed waiting on `curl` subprocess completion")?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!(
                "`curl` request to {} failed with status {}: {}",
                endpoint,
                output.status,
                stderr.trim()
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (body_part, code_part) = stdout.rsplit_once('\n').unwrap_or((&stdout, "200"));
        let http_status: u16 = code_part.trim().parse().unwrap_or(200);
        parse_openai_chat_completion_response(body_part, http_status)
    }
}

/// Format a [`ChatMessage`] conversation slice as a flattened plain-text transcript for
/// non-interactive `codex exec` stdin (`-`), prepending the strict no-shell-command instruction.
#[must_use]
pub fn format_codex_exec_stdin_prompt(messages: &[ChatMessage]) -> String {
    let mut out =
        String::from("Do NOT run shell commands; reply directly in plain text as instructed.\n\n");
    for (idx, msg) in messages.iter().enumerate() {
        if idx > 0 {
            out.push_str("\n\n");
        }
        let role_tag = match msg.role.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => "[SYSTEM]",
            "assistant" => "[ASSISTANT]",
            _ => "[USER]",
        };
        out.push_str(role_tag);
        out.push('\n');
        out.push_str(msg.content.trim());
    }
    out.push('\n');
    out
}

fn normalize_optional_codex_model_override(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("codex-default"))
        .map(ToOwned::to_owned)
}

/// Flat-rate subscription OpenAI transport invoking `codex exec --ephemeral --skip-git-repo-check
/// -s read-only -C <isolated_empty_tempdir> -o <output_last_message_file> -` with zero `$OPENAI_API_KEY` required.
#[derive(Debug, Clone)]
pub struct CodexExecChatTransport {
    codex_bin: String,
    model_override: Option<String>,
    timeout: Duration,
}

impl Default for CodexExecChatTransport {
    fn default() -> Self {
        Self::from_env(None)
    }
}

impl CodexExecChatTransport {
    /// Construct a [`CodexExecChatTransport`] with explicit binary path, optional custom `-m <model>`
    /// override (`None` or `"codex-default"` omits `-m` so `codex` uses its configured flagship model),
    /// and subprocess timeout.
    pub fn new(
        codex_bin: impl Into<String>,
        model_override: Option<String>,
        timeout: Duration,
    ) -> Self {
        let raw_bin = codex_bin.into();
        let trimmed_bin = raw_bin.trim();
        let codex_bin = if trimmed_bin.is_empty() {
            DEFAULT_CODEX_BIN.to_string()
        } else {
            trimmed_bin.to_string()
        };
        Self {
            codex_bin,
            model_override: normalize_optional_codex_model_override(model_override.as_deref()),
            timeout: timeout.max(Duration::from_millis(50)),
        }
    }

    /// Construct a [`CodexExecChatTransport`] honoring `$PORTIA_CODEX_BIN` (default `"codex"`)
    /// and optional explicit `openai_model` or `$OPENAI_SYNTH_MODEL` override.
    pub fn from_env(openai_model: Option<&str>) -> Self {
        let codex_bin = std::env::var("PORTIA_CODEX_BIN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_CODEX_BIN.to_string());
        let env_model = std::env::var("OPENAI_SYNTH_MODEL")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let explicit_model = normalize_optional_codex_model_override(openai_model)
            .or_else(|| normalize_optional_codex_model_override(env_model.as_deref()));
        Self::new(
            codex_bin,
            explicit_model,
            Duration::from_secs(DEFAULT_CODEX_TIMEOUT_SECS),
        )
    }

    /// Construct a [`CodexExecChatTransport`] for `--synth-user luna`, defaulting `-m` to
    /// [`DEFAULT_CODEX_LUNA_MODEL`] (`"gpt-6-luna"`) unless `openai_model` overrides it.
    pub fn from_luna_env(openai_model: Option<&str>) -> Self {
        let codex_bin = std::env::var("PORTIA_CODEX_BIN")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_CODEX_BIN.to_string());
        let resolved_model = normalize_optional_codex_model_override(openai_model)
            .unwrap_or_else(|| DEFAULT_CODEX_LUNA_MODEL.to_string());
        Self::new(
            codex_bin,
            Some(resolved_model),
            Duration::from_secs(DEFAULT_CODEX_TIMEOUT_SECS),
        )
    }

    pub fn with_codex_bin(mut self, codex_bin: impl Into<String>) -> Self {
        let trimmed = codex_bin.into().trim().to_string();
        if !trimmed.is_empty() {
            self.codex_bin = trimmed;
        }
        self
    }

    pub fn with_model_override(mut self, model_override: Option<String>) -> Self {
        self.model_override = normalize_optional_codex_model_override(model_override.as_deref());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout.max(Duration::from_millis(50));
        self
    }

    pub fn codex_bin(&self) -> &str {
        &self.codex_bin
    }

    pub fn model_override(&self) -> Option<&str> {
        self.model_override.as_deref()
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }
}

impl ChatCompletionTransport for CodexExecChatTransport {
    fn complete_chat(&mut self, req: &ChatCompletionRequest) -> Result<String> {
        self.complete_chat_with_schema(req, None)
    }

    fn complete_chat_with_schema(
        &mut self,
        req: &ChatCompletionRequest,
        output_schema: Option<&serde_json::Value>,
    ) -> Result<String> {
        let isolated_dir = tempfile::Builder::new()
            .prefix("portia_codex_empty_")
            .tempdir()
            .context("failed to create isolated empty tempdir for `codex exec`")?;
        let out_file = tempfile::Builder::new()
            .prefix("portia_codex_last_msg_")
            .suffix(".txt")
            .tempfile()
            .context("failed to create output file for `codex exec -o`")?;
        let schema_file = if let Some(schema_val) = output_schema {
            let mut sf = tempfile::Builder::new()
                .prefix("portia_codex_schema_")
                .suffix(".json")
                .tempfile()
                .context("failed to create JSON schema file for `codex exec --output-schema`")?;
            let schema_bytes = serde_json::to_vec_pretty(schema_val)
                .context("failed to serialize JSON schema for `codex exec --output-schema`")?;
            sf.write_all(&schema_bytes)
                .context("failed writing JSON schema file for `codex exec --output-schema`")?;
            sf.flush()?;
            Some(sf)
        } else {
            None
        };

        let effective_model = self.model_override.as_deref().or_else(|| {
            let m = req.model.trim();
            if m.is_empty() || m.eq_ignore_ascii_case("codex-default") {
                None
            } else {
                Some(m)
            }
        });

        let stdin_prompt = format_codex_exec_stdin_prompt(&req.messages);
        let mut cmd = Command::new(&self.codex_bin);
        cmd.arg("exec")
            .arg("--ephemeral")
            .arg("--skip-git-repo-check")
            .arg("-s")
            .arg("read-only")
            .arg("-C")
            .arg(isolated_dir.path());
        if let Some(model_id) = effective_model {
            cmd.arg("-m").arg(model_id);
        }
        if let Some(ref sf) = schema_file {
            cmd.arg("--output-schema").arg(sf.path());
        }
        cmd.arg("-o")
            .arg(out_file.path())
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "failed to spawn `{}` for CodexExecChatTransport",
                self.codex_bin
            )
        })?;

        {
            let mut stdin = child
                .stdin
                .take()
                .context("failed to open stdin pipe to `codex exec`")?;
            stdin
                .write_all(stdin_prompt.as_bytes())
                .context("failed to write flattened prompt to `codex exec` stdin")?;
        }

        let mut child_stdout = child.stdout.take();
        let mut child_stderr = child.stderr.take();
        let stdout_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(ref mut out) = child_stdout {
                let _ = out.read_to_end(&mut buf);
            }
            buf
        });
        let stderr_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(ref mut err) = child_stderr {
                let _ = err.read_to_end(&mut buf);
            }
            buf
        });

        let deadline = Instant::now() + self.timeout;
        let exit_status = loop {
            match child
                .try_wait()
                .context("failed polling `codex exec` subprocess")?
            {
                Some(status) => break status,
                None => {
                    if Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        bail!(
                            "`codex exec` ({}) timed out after {:?}",
                            self.codex_bin,
                            self.timeout
                        );
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        };

        let stdout_bytes = stdout_handle.join().unwrap_or_default();
        let stderr_bytes = stderr_handle.join().unwrap_or_default();
        if !exit_status.success() {
            let stderr_txt = String::from_utf8_lossy(&stderr_bytes);
            let stdout_txt = String::from_utf8_lossy(&stdout_bytes);
            bail!(
                "`codex exec` ({}) exited with {}: stderr={} stdout={}",
                self.codex_bin,
                exit_status,
                stderr_txt.trim(),
                stdout_txt.trim()
            );
        }

        let raw_msg = std::fs::read_to_string(out_file.path())
            .context("failed reading `codex exec -o` output message file")?;
        let trimmed = raw_msg.trim().to_string();
        if trimmed.is_empty() {
            let stderr_txt = String::from_utf8_lossy(&stderr_bytes);
            bail!(
                "`codex exec` wrote empty output to `-o` message file (stderr={})",
                stderr_txt.trim()
            );
        }
        Ok(trimmed)
    }
}

/// Auto-selecting OpenAI chat transport dispatching to [`CurlChatCompletionTransport`] when
/// `$OPENAI_API_KEY` is present and falling back automatically to [`CodexExecChatTransport`]
/// (`codex exec`) on flat-rate subscription hosts with no API key configured.
#[derive(Debug, Clone)]
pub enum AutoOpenAiTransport {
    ApiCurl(CurlChatCompletionTransport),
    CodexSubscription(CodexExecChatTransport),
}

impl AutoOpenAiTransport {
    /// Resolve [`AutoOpenAiTransport::ApiCurl`] when `$OPENAI_API_KEY` is non-empty in the
    /// environment, or [`AutoOpenAiTransport::CodexSubscription`] otherwise.
    pub fn from_env(openai_model: Option<&str>) -> Result<Self> {
        let has_api_key = std::env::var("OPENAI_API_KEY")
            .ok()
            .is_some_and(|v| !v.trim().is_empty());
        if has_api_key {
            let mut curl = CurlChatCompletionTransport::from_env()?;
            if let Some(model) = openai_model.map(str::trim).filter(|s| !s.is_empty()) {
                curl = curl.with_model(model);
            }
            Ok(Self::ApiCurl(curl))
        } else {
            Ok(Self::CodexSubscription(CodexExecChatTransport::from_env(
                openai_model,
            )))
        }
    }

    /// Return the request model identifier to pair with [`OpenAiChatSynthUserEnv::new`]
    /// (`"codex-default"` when running `CodexSubscription` without an explicit `-m` override).
    #[must_use]
    pub fn effective_model_label(&self) -> &str {
        match self {
            Self::ApiCurl(curl) => curl.model(),
            Self::CodexSubscription(codex) => codex.model_override().unwrap_or("codex-default"),
        }
    }
}

impl ChatCompletionTransport for AutoOpenAiTransport {
    fn complete_chat(&mut self, req: &ChatCompletionRequest) -> Result<String> {
        match self {
            Self::ApiCurl(curl) => curl.complete_chat(req),
            Self::CodexSubscription(codex) => codex.complete_chat(req),
        }
    }

    fn complete_chat_with_schema(
        &mut self,
        req: &ChatCompletionRequest,
        output_schema: Option<&serde_json::Value>,
    ) -> Result<String> {
        match self {
            Self::ApiCurl(curl) => curl.complete_chat_with_schema(req, output_schema),
            Self::CodexSubscription(codex) => codex.complete_chat_with_schema(req, output_schema),
        }
    }
}
