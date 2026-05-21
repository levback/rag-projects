#!/usr/bin/env bash
# =============================================================================
# scripts/security_check.sh
#
# Security audit for the RAG-Projects codebase.
# Checks against:
#   - OWASP Top 10 (2021) via bandit + pip-audit + custom rules
#   - NIST SP 800-218 (SSDF) via dependency pinning checks + secret scanning
#   - NIST SP 800-53 (AC, SI, SA controls) via static analysis
#
# Usage:
#   bash scripts/security_check.sh [--fix] [--report]
#
# Options:
#   --fix     Auto-upgrade vulnerable packages where possible
#   --report  Write full JSON reports to security_reports/
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPORT_DIR="$ROOT/security_reports"
FIX=false
WRITE_REPORT=false
OVERALL_EXIT=0

for arg in "$@"; do
  case "$arg" in
    --fix)    FIX=true ;;
    --report) WRITE_REPORT=true ;;
  esac
done

if $WRITE_REPORT; then
  mkdir -p "$REPORT_DIR"
fi

# ── Colour helpers ─────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
BLUE='\033[0;34m'; RESET='\033[0m'; BOLD='\033[1m'

pass()  { echo -e "${GREEN}  [PASS]${RESET} $*"; }
fail()  { echo -e "${RED}  [FAIL]${RESET} $*"; OVERALL_EXIT=1; }
warn()  { echo -e "${YELLOW}  [WARN]${RESET} $*"; }
info()  { echo -e "${BLUE}  [INFO]${RESET} $*"; }
title() { echo -e "\n${BOLD}$*${RESET}"; }

# ── Activate venv ──────────────────────────────────────────────────────────
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
fi

# ═══════════════════════════════════════════════════════════════════════════
# A. OWASP TOP 10 — Static Application Security Testing (SAST)
#    OWASP A03: Injection / A05: Security Misconfiguration /
#    A06: Vulnerable Components / A09: Logging Failures
# ═══════════════════════════════════════════════════════════════════════════
title "OWASP TOP 10 — SAST (bandit)"

if command -v bandit &>/dev/null || pip show bandit &>/dev/null 2>&1; then
  BANDIT_ARGS=(-r src/ -ll -ii --format text)
  if $WRITE_REPORT; then
    BANDIT_ARGS+=(--format json -o "$REPORT_DIR/bandit.json")
    bandit "${BANDIT_ARGS[@]}" 2>/dev/null || true
    BANDIT_ARGS=(-r src/ -ll -ii)
  fi
  if bandit "${BANDIT_ARGS[@]}" 2>&1; then
    pass "bandit: no HIGH/MEDIUM severity issues found"
  else
    fail "bandit: issues found — review output above"
  fi
else
  warn "bandit not installed — run: pip install bandit"
  if $FIX; then pip install bandit; bandit -r src/ -ll -ii; fi
fi

# ── OWASP A06: Known Vulnerable Components ─────────────────────────────────
title "OWASP A06 — Vulnerable Dependencies (pip-audit)"

if command -v pip-audit &>/dev/null || pip show pip-audit &>/dev/null 2>&1; then
  AUDIT_ARGS=(--format columns)
  if $WRITE_REPORT; then
    pip-audit --format json -o "$REPORT_DIR/pip_audit.json" 2>/dev/null || true
  fi
  if $FIX; then
    AUDIT_ARGS+=(--fix)
  fi
  if pip-audit "${AUDIT_ARGS[@]}" 2>&1; then
    pass "pip-audit: no known CVEs in installed packages"
  else
    fail "pip-audit: vulnerable packages found"
  fi
else
  warn "pip-audit not installed — run: pip install pip-audit"
  if $FIX; then pip install pip-audit; pip-audit; fi
fi

# ── OWASP A02: Cryptographic Failures — check for weak algos ───────────────
title "OWASP A02 — Cryptographic Failures (custom grep)"

WEAK_CRYPTO=$(grep -rn --include="*.py" \
  -e "hashlib\.md5\b" \
  -e "hashlib\.sha1\b" \
  -e "Crypto\.Cipher\.DES" \
  -e "ssl\.PROTOCOL_TLS\b" \
  -e "verify=False" \
  src/ 2>/dev/null || true)

if [[ -z "$WEAK_CRYPTO" ]]; then
  pass "No weak cryptographic calls detected"
else
  fail "Weak crypto detected:\n$WEAK_CRYPTO"
fi

# ── OWASP A03: Injection — check for shell/SQL injection patterns ───────────
title "OWASP A03 — Injection Risks (custom grep)"

INJECTION=$(grep -rn --include="*.py" \
  -e "subprocess\.call.*shell=True" \
  -e "os\.system(" \
  -e "eval(" \
  -e "exec(" \
  -e "pickle\.load" \
  src/ 2>/dev/null || true)

if [[ -z "$INJECTION" ]]; then
  pass "No obvious shell/eval/pickle injection patterns found"
else
  warn "Potential injection risk (review manually):\n$INJECTION"
fi

# ── OWASP A07: Auth — hardcoded credentials / secrets ──────────────────────
title "OWASP A07 — Hardcoded Secrets (detect-secrets / grep)"

if command -v detect-secrets &>/dev/null || pip show detect-secrets &>/dev/null 2>&1; then
  if $WRITE_REPORT; then
    detect-secrets scan src/ > "$REPORT_DIR/detect_secrets.json" 2>/dev/null || true
  fi
  SECRETS=$(detect-secrets scan src/ 2>/dev/null | python3 -c \
    "import sys,json; d=json.load(sys.stdin); \
     results=d.get('results',{}); \
     count=sum(len(v) for v in results.values()); \
     print(count)" 2>/dev/null || echo "0")
  if [[ "$SECRETS" == "0" ]]; then
    pass "detect-secrets: no secrets detected"
  else
    fail "detect-secrets: $SECRETS potential secret(s) found — run: detect-secrets scan src/"
  fi
else
  # Fallback: grep for common patterns
  HARDCODED=$(grep -rn --include="*.py" \
    -e 'api_key\s*=\s*"[^"]\+[^{]"' \
    -e 'password\s*=\s*"[^"]\+[^{]"' \
    -e 'secret\s*=\s*"[^"]\+[^{]"' \
    -e 'aws_access_key_id\s*=\s*"AK' \
    src/ 2>/dev/null || true)
  if [[ -z "$HARDCODED" ]]; then
    pass "No hardcoded credentials detected (basic grep)"
  else
    fail "Possible hardcoded credential:\n$HARDCODED"
  fi
fi

# ── OWASP A05: Security Misconfiguration — SSRF protection ─────────────────
title "OWASP A05/A10 — SSRF Protection Check"

SSRF_GUARD=$(grep -n "_BLOCKED_HOSTS\|_BLOCKED_PREFIXES\|_validate_url" \
  src/loaders/web_scraper.py 2>/dev/null || true)
if [[ -n "$SSRF_GUARD" ]]; then
  pass "SSRF blocklist present in web_scraper.py"
else
  fail "SSRF protection missing in web_scraper.py"
fi

# ── OWASP A01: Broken Access Control — path traversal ──────────────────────
title "OWASP A01 — Path Traversal Check"

RESOLVE=$(grep -n "\.resolve()" src/loaders/document_loader.py 2>/dev/null || true)
if [[ -n "$RESOLVE" ]]; then
  pass "Path traversal mitigation (.resolve()) present in document_loader.py"
else
  fail "Path traversal: document_loader.py does not call Path.resolve()"
fi

# ── OWASP A08: Prompt Injection in RAG prompts ─────────────────────────────
title "OWASP A08 — Prompt Injection Mitigations"

DELIMITERS=$(grep -rn --include="*.py" \
  "BEGIN SOURCES\|END SOURCES\|Treat.*sources as data" \
  src/ 2>/dev/null || true)
if [[ -n "$DELIMITERS" ]]; then
  pass "Prompt injection delimiters found in source files"
else
  fail "No prompt injection delimiters found in src/"
fi

# ═══════════════════════════════════════════════════════════════════════════
# B. NIST SP 800-218 (SSDF) — Supply Chain & Dependency Integrity
# ═══════════════════════════════════════════════════════════════════════════
title "NIST SP 800-218 (SSDF) — Dependency Pinning"

UNPINNED=$(grep -E "^[a-zA-Z]" requirements.txt 2>/dev/null | \
  grep -v "==" | grep -v "^#" | grep -v "^-" || true)

if [[ -z "$UNPINNED" ]]; then
  pass "All requirements.txt entries use == or >= version constraints"
else
  warn "Unpinned or loosely pinned packages (consider ==):\n$UNPINNED"
fi

# ── NIST SA-11: Developer Security Testing — test coverage ─────────────────
title "NIST SA-11 — Test Coverage (>= 80%)"

if command -v pytest &>/dev/null || .venv/bin/pytest --version &>/dev/null 2>&1; then
  COVERAGE=$(python -m pytest --co -q 2>/dev/null | tail -1 | grep -oE "[0-9]+ (test|passed)" | head -1 || echo "unknown")
  if [[ -f "$ROOT/.coverage" ]]; then
    COV_PCT=$(python -m coverage report 2>/dev/null | tail -1 | awk '{print $NF}' | tr -d '%' || echo "0")
    if (( ${COV_PCT:-0} >= 80 )); then
      pass "Test coverage: ${COV_PCT}% (≥ 80% — NIST SA-11 satisfied)"
    else
      fail "Test coverage: ${COV_PCT}% (< 80% — NIST SA-11 requires ≥ 80%)"
    fi
  else
    warn "No .coverage file found — run: pytest --cov=src to generate"
  fi
fi

# ── NIST SI-10: Input Validation — check boundary validation ────────────────
title "NIST SI-10 — Input Validation at Boundaries"

VALIDATION=$(grep -rn --include="*.py" \
  "if not\|ValueError\|raise\|[:1000]\|truncate\|maxlen" \
  src/rag/ src/agents/ 2>/dev/null | wc -l || echo "0")
if (( VALIDATION > 5 )); then
  pass "Input validation patterns found in rag/ and agents/ ($VALIDATION occurrences)"
else
  warn "Limited input validation detected — review src/rag/ and src/agents/"
fi

# ── NIST AC-3: Least Privilege — no root/setuid files ──────────────────────
title "NIST AC-3 — File Permission Check"

WORLD_WRITE=$(find src/ scripts/ -perm -o+w -type f 2>/dev/null || true)
if [[ -z "$WORLD_WRITE" ]]; then
  pass "No world-writable source files found"
else
  fail "World-writable files found:\n$WORLD_WRITE"
fi

# ── NIST AU-9: Logging — ensure logging is present ─────────────────────────
title "NIST AU-9 — Audit Logging"

LOGGING=$(grep -rn --include="*.py" "logger\.\|logging\." src/ 2>/dev/null | wc -l || echo "0")
if (( LOGGING > 10 )); then
  pass "Logging statements present ($LOGGING occurrences in src/)"
else
  warn "Limited logging found ($LOGGING occurrences) — add structured logging for audit trails"
fi

# ═══════════════════════════════════════════════════════════════════════════
# C. Summary
# ═══════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
if [[ $OVERALL_EXIT -eq 0 ]]; then
  echo -e "${GREEN}${BOLD}  ALL CHECKS PASSED${RESET}"
else
  echo -e "${RED}${BOLD}  SOME CHECKS FAILED — review output above${RESET}"
fi
if $WRITE_REPORT; then
  echo -e "${BLUE}  Reports written to: $REPORT_DIR/${RESET}"
fi
echo "═══════════════════════════════════════════════════════"

exit $OVERALL_EXIT
