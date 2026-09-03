# Security Policy

> ⚠️ **This project involves code execution (Triton kernels are compiled and run on real GPUs).** Unaudited kernel code, untrusted prompts, or externally injected inputs can trigger arbitrary code execution, illegal memory access, or resource exhaustion. Treat any kernel and any input produced by this project as **untrusted**, and execute only in isolated environments.

## Supported Versions

| Version | Status |
|------|----------|
| 0.1.x | ✅ actively maintained |
| other | ❌ unsupported |

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities. Report privately via:

- **Email**: `2622507532@qq.com`
- **GitHub private vulnerability disclosure**: use GitHub's [Security Advisory feature](https://docs.github.com/en/code-security/security-advisories) (if enabled) to file a private report.

Please include, as much as possible:

1. Vulnerability type and impact scope (remotely triggerable? arbitrary code execution?).
2. Reproduction path (minimal triggering kernel / input / configuration).
3. Affected modules and versions.
4. If you have confirmed a kernel or input can execute arbitrary code without authorization, prefer private disclosure and **pause public sharing** until fixed.

## Response Commitments

| Stage | Target |
|----------|------|
| Acknowledge report | within 5 business days |
| Initial triage & severity assessment (CVSS) | within 10 business days |
| Fix release (critical/high) | within 30 days (extendable with progress updates) |
| Fix release (medium) | within 60 days |
| Low / best-practice advisories | folded into regular releases, no separate SLA |

Confirmed vulnerabilities are **not publicly disclosed** before the fix is released, giving users time to upgrade. Security advisories are published after fixes as needed.

## Project-Specific Notes (code-execution surface)

This project is a kernel-generation RLVR pipeline; its security surface differs from an ordinary library:

- **The evaluation environment (`kernelgym/`)** forks subprocess pools to compile and execute generated kernels. Any Triton code from model / external input runs inside this environment — please ensure:
  - The evaluation / grading server runs in a **restricted container or VM** without sensitive host privileges.
  - The grading server port (default 10907) is not exposed to **untrusted networks**.
- **The training side (`drkernel/` verl integration)** feeds model output directly into the execution environment — a high-risk path subject to the same isolation requirements.
- If you discover a path enabling host escape, privilege escalation, or unexpected file/network access from inside a kernel, report it at **critical** priority.

## Handling Process

1. Maintainers acknowledge and reply to the report.
2. Severity and impact are assessed; a fix plan is drafted.
3. The fix and regression tests are developed and verified on a private branch.
4. The fix and a security advisory are released, disclosing affected versions and upgrade guidance.

Thanks for helping make King.triton-kernel safer for everyone.
