# Summary

<!-- One sentence describing what changed and why. -->

## Changes

<!-- Bulleted list of the key changes. -->

## Test plan

- [ ] `cargo test --workspace` passes
- [ ] `pytest` passes locally (with `maturin develop --release` rebuilt)
- [ ] Property tests added (if new public API)
- [ ] Differential test added or updated (if numerical change)
- [ ] Type stubs updated (if `_core` surface changed)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] py_vollib compat preserved (`tests/test_compat.py` still green)

## Conventional commit

This PR uses a conventional commit title (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `perf:`, `test:`) so release-please picks it up correctly.

## Related issues

<!-- Closes #N -->
