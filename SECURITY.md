# Security Policy

## Supported Versions

Security fixes are applied to the latest minor release of each crate:

| Crate | Supported Version |
| :--- | :---: |
| `vxg` | `0.7.x` |
| `vxg-kv` | `0.3.x` |
| `vxg-tensor` | `0.6.x` |
| `vimg` | `0.4.x` |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities (such as memory unsafety, untrusted-input panics/OOMs, authentication bypass in `vxg-kv`, or path traversal in `vxg serve` / `vxg-kv origin`).

Instead, report vulnerabilities privately via **GitHub Private Vulnerability Reporting** (`Security` → `Advisories` → `Report a vulnerability` on [meftunca/vxg](https://github.com/meftunca/vxg/security)).

Please include:
- Affected crate and version
- Minimal reproduction input or proof-of-concept
- Expected vs. actual behavior

We aim to acknowledge reports within **48 hours** and publish a fix + regression test as soon as verified.

For our untrusted-input memory caps, CRC32C integrity guarantees, and fuzzing harness details, see [`docs/SAFETY.md`](docs/SAFETY.md).
