# Security Policy

## Supported Versions

| Version | Supported          | Security Maintenance |
| ------- | ------------------ | -------------------- |
| 1.0.x   | :white_check_mark: | Active Support       |
| < 1.0   | :x:                | End of Life          |

## Reporting a Vulnerability

The Vajra-AI security team takes security vulnerabilities seriously.

If you discover a security issue or vulnerability in `vajra` (including `vajra-lm` and `vajra-agent`), please report it privately:

- **Email**: `security@vajra-ai.org`
- **GPG Key**: Available upon request for encrypted communications.

### Disclosure Process
1. **Do NOT open a public GitHub issue** or disclose details publicly until the vulnerability has been triaged and addressed.
2. Provide detailed steps to reproduce the vulnerability, including code snippets or proof-of-concept exploits if available.
3. The Vajra security team will acknowledge receipt within **48 hours** and provide periodic updates on progress towards resolution.
4. Once a fix is released, public advisories and CVE requests will be coordinated transparently.

## Security Scope

The scope of this security policy covers:
- Core model architecture & execution (`model/`)
- Agent execution sandboxes (`vajra_agent/sandbox/`, `vajra_agent/tools/`)
- API endpoints (`api/`)
- Dependency configuration & Docker containerization (`Dockerfile`, `Dockerfile.serve`)
