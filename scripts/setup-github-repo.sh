#!/usr/bin/env bash
# scripts/setup-github-repo.sh
#
# One-shot replayable setup of the GitHub-side configuration for pyvolr:
#   - Topics + description
#   - Dependabot security alerts + automated security fixes
#   - Private vulnerability reporting
#   - Minimal branch protection on main (linear history, no force push,
#     no deletions, conversation resolution required)
#
# This DOES NOT create the repo (that's a one-time `gh repo create` -- see
# the README "Going public" section). All commands here are idempotent and
# safe to re-run after a fresh fork or transfer.
#
# Required: `gh` CLI authenticated; repo already exists at OWNER/REPO.

set -euo pipefail

OWNER="${OWNER:-yipjunkai}"
REPO="${REPO:-pyvolr}"
SLUG="$OWNER/$REPO"

echo "==> Configuring $SLUG"

echo "==> Setting topics + description"
gh repo edit "$SLUG" \
    --description "Modern Black-Scholes-Merton pricing, Greeks, and implied volatility for Python. Rust core. Drop-in replacement for the abandoned py_vollib." \
    --homepage "https://github.com/$SLUG" \
    --add-topic python \
    --add-topic rust \
    --add-topic black-scholes \
    --add-topic black-scholes-merton \
    --add-topic options \
    --add-topic options-pricing \
    --add-topic quantitative-finance \
    --add-topic quant \
    --add-topic greeks \
    --add-topic implied-volatility \
    --add-topic pyo3 \
    --add-topic maturin \
    --add-topic numpy \
    --add-topic py-vollib

echo "==> Enabling Dependabot vulnerability alerts"
gh api -X PUT "/repos/$SLUG/vulnerability-alerts" --silent

echo "==> Enabling automated security fixes"
gh api -X PUT "/repos/$SLUG/automated-security-fixes" --silent

echo "==> Enabling private vulnerability reporting"
gh api -X PUT "/repos/$SLUG/private-vulnerability-reporting" --silent

echo "==> Applying minimal branch protection on main"
# Solo-maintainer baseline:
#   - linear history (no merge commits)
#   - no force pushes
#   - no branch deletion
#   - conversation resolution required before merge
#   - admins not enforced (so the maintainer can hotfix in a pinch)
#   - NO required PR reviewer count (raise to 1+ once a co-maintainer exists)
#   - NO required status check contexts (add after first CI run reveals exact
#     check names; see scripts/tighten-branch-protection.sh -- TODO)
gh api -X PUT "/repos/$SLUG/branches/main/protection" \
    -H "Accept: application/vnd.github+json" \
    -F 'enforce_admins=false' \
    -F 'required_pull_request_reviews=null' \
    -F 'required_status_checks=null' \
    -F 'restrictions=null' \
    -F 'required_linear_history=true' \
    -F 'allow_force_pushes=false' \
    -F 'allow_deletions=false' \
    -F 'required_conversation_resolution=true' \
    --silent

echo
echo "==> Done. Manual follow-ups (web UI only):"
echo "    1. Verify 2FA active at https://github.com/settings/security"
echo "    2. Enable GitHub Pages (Settings -> Pages -> Source: GitHub Actions)"
echo "       so the docs.yml workflow can deploy."
echo "    3. After first CI run, optionally tighten branch protection to"
echo "       require specific status checks (use the names shown in the"
echo "       Actions tab as required_status_checks contexts)."
echo "    4. Configure PyPI Trusted Publishing at"
echo "       https://pypi.org/manage/account/publishing/ once you reserve"
echo "       the project name (binding: repo=$SLUG, workflow=release.yml,"
echo "       environment=pypi)."
