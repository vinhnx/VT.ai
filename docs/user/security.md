# Security Information

This page provides information about security vulnerabilities and updates in VT.ai.

## Latest Security Release: v0.7.5

**Release Date:** April 2, 2026

**Severity:** Critical

VT.ai v0.7.5 addresses **25 security vulnerabilities** in third-party dependencies. All users should upgrade immediately.

### Vulnerabilities Fixed

#### High Severity

| Package | CVE | Vulnerability | Impact | Fixed Version |
|---------|-----|---------------|--------|---------------|
| **aiohttp** | CVE-2026-34525 | Multiple Host headers accepted | Security bypass | 3.13.5 |
| **aiohttp** | CVE-2026-34520 | Control characters in headers | Header injection | 3.13.5 |
| **aiohttp** | CVE-2026-34519 | HTTP response splitting | Response splitting | 3.13.5 |
| **aiohttp** | CVE-2026-34517 | Late size enforcement | DoS via memory exhaustion | 3.13.5 |
| **aiohttp** | CVE-2026-34518 | Cookie header leak on redirect | Information disclosure | 3.13.5 |
| **cryptography** | CVE-2026-26007 | SECT curve subgroup validation bypass | Cryptographic weakness | 46.0.6 |
| **mcp** | CVE-2025-66416 | DNS rebinding protection disabled by default | Local network access | 1.26.0 |
| **PyJWT** | CVE-2026-32597 | Critical header (`crit`) validation bypass | Token security bypass | 2.12.1 |
| **black** | CVE-2026-32274 | Path traversal via `--python-cell-magics` | Arbitrary file write | 26.3.1 |
| **onnx** | GHSA-q56x-g2fj-4rj6 | TOCTOU race condition | Arbitrary file read/write | 1.21.0 |
| **pillow** | CVE-2026-25990 | Out-of-bounds write in PSD parser | Code execution | 12.2.0 |

### How to Upgrade

#### PyPI Installation

```bash
pip install --upgrade vtai
```

#### uv Installation

```bash
uv pip install --upgrade vtai
```

#### Verify Installation

After upgrading, verify your installation:

```bash
pip show vtai
```

You should see version `0.7.5` or later.

### Technical Details

#### aiohttp Vulnerabilities

The aiohttp vulnerabilities affect HTTP request/response handling:

- **Multiple Host Headers**: Could allow security bypass in reverse proxy configurations
- **Header Injection**: Control characters in header values could lead to request smuggling
- **Response Splitting**: Unvalidated reason phrases could enable response splitting attacks
- **Memory DoS**: Delayed size checking for multipart fields
- **Header Leak**: Cookie and Proxy-Authorization headers leaked on cross-origin redirects

**Reference:** [aiohttp Security Advisories](https://github.com/aio-libs/aiohttp/security/advisories)

#### cryptography Vulnerability

The SECT curve vulnerability affects elliptic curve cryptography:

- Missing subgroup validation for SECT curves
- Could leak private key information via small subgroup attacks
- Affects ECDSA signature verification and ECDH key exchange

**Reference:** [cryptography Advisory](https://github.com/pyca/cryptography/security/advisories/GHSA-r6ph-v2qm-q3c2)

#### PyJWT Vulnerability

The `crit` header validation bypass affects JWT security:

- Critical header extensions not properly validated
- Could allow token binding bypass
- May enable security policy circumvention

**Reference:** [PyJWT Advisory](https://github.com/jpadilla/pyjwt/security/advisories/GHSA-752w-5fwx-jx9f)

#### pillow Vulnerability

The PSD parser vulnerability:

- Out-of-bounds write when loading crafted PSD images
- Could lead to arbitrary code execution
- Affects Pillow versions >= 10.3.0, < 12.1.1

**Reference:** [pillow Advisory](https://github.com/python-pillow/Pillow/security/advisories/GHSA-cfh3-3jmp-rvhc)

### Reporting Security Issues

We take security seriously. If you discover a security vulnerability in VT.ai, please report it responsibly:

1. **Do not** create a public GitHub issue
2. Email: [security contact]
3. Include detailed reproduction steps
4. Allow reasonable time for response and fix

### Security Best Practices

When using VT.ai, follow these security best practices:

1. **Keep Updated**: Always use the latest version of VT.ai
2. **API Key Security**: Store API keys securely in environment variables or `.env` files
3. **Access Control**: Run VT.ai in a secure, access-controlled environment
4. **Network Security**: Use HTTPS and secure network configurations
5. **Monitor Dependencies**: Watch for security updates in transitive dependencies

### Previous Security Releases

- **v0.7.4** and earlier: No known critical vulnerabilities at time of release

---

**Last Updated:** April 2, 2026

**Version:** 0.7.5
