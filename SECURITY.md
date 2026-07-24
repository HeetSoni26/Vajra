# Security Policy

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `v1.0.x` | :white_check_mark: Supported |
| `< 1.0.0` | :x: Unsupported |

---

## Reporting a Vulnerability

The Vajra AI team takes security and model safety seriously. If you discover a security vulnerability (such as unsafe deserialization, arbitrary code execution, or data exposure), please follow these steps:

1. **Do NOT open a public GitHub issue.**
2. Report the vulnerability directly to `security@vajra.ai` or submit a private security advisory via GitHub Security Advisories.
3. Include detailed steps to reproduce the issue, proof-of-concept scripts, and affected versions.

---

## Security Best Practices
- **Safe Weight Deserialization**: Always use `model.safetensors` over `pytorch_model.bin` when loading untrusted weights to prevent pickle execution vulnerabilities.
- **Model Alignment**: Base pretraining checkpoints are raw foundation models without RLHF or instruction tuning safety filters. Deployments should implement downstream guardrails.
