# Security policy

Security fixes are made on `main`. Use the latest release when one is available.
This is a personal portfolio project, without a guaranteed response time or security certification.

## Report a vulnerability privately

Open the repository's **Security** tab and choose **Report a vulnerability**:
https://github.com/mustafaalazzawi1/pytorch-digit-classifier/security/advisories/new

Include the affected commit or version, reproduction steps, expected and actual behavior,
and the potential impact. Use synthetic examples and redact passwords, tokens and personal data.
Please do not disclose an unpatched vulnerability or secret in a public issue.

Test only systems and data you own or are authorized to assess. Do not probe the hosted
portfolio or demo with destructive tests, traffic floods, or real credentials.

Model exports use validated JSON. Do not load untrusted pickle-based checkpoints or add unsafe deserialization.
