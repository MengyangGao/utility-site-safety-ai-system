# Security policy

Security fixes are applied to the latest release and the default branch.

## Report a vulnerability

Use GitHub's private **Report a vulnerability** / Security Advisory flow. Please do not open a
public issue for exposed camera credentials, unsafe model loading, path traversal, command injection,
privacy-redaction bypasses, session isolation problems, or reports that expose private data.

Include the affected version, a minimal reproduction, impact, and sanitized logs. Never attach real
worksite footage, credentials, private model weights, or personal data.

## Safe operation

- Keep the Web console on a local or access-controlled network.
- Keep privacy redaction enabled and review media before sharing it.
- Protect `outputs/`, camera URLs, tokens, and report downloads.
- Load only trusted model files; PyTorch `.pt` files can contain executable code.
- Update Python, PyTorch, Ultralytics, Streamlit, and OpenCV regularly.
