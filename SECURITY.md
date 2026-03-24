# Security Policy

## LiteLLM Supply Chain Attack (March 2026)

### Affected Versions
- **litellm 1.82.7** - Malicious
- **litellm 1.82.8** - Malicious

These versions contain a malicious `.pth` file (`litellm_init.pth`) that harvests SSH keys, cloud credentials, and secrets on every Python startup.

### Safe Versions
- **litellm >=1.80.0, <1.82.7** - Safe (including our current version 1.80.0)
- Future versions should be verified against official BerriAI announcements

### Security Audit Completed (March 25, 2026)

**NOT AFFECTED** - Full audit completed:

| Check | Status | Details |
|-------|--------|---------|
| Installed version | Safe | litellm 1.80.0 (both system and .venv) |
| uv cache | Clean | No litellm files in ~/.cache/uv |
| Malicious .pth files | None | No litellm_init.pth found anywhere |
| sysmon persistence | None | ~/.config/sysmon/ does not exist |
| systemd service | None | No sysmon.service found |
| Shell config files | Clean | No suspicious entries in .bashrc/.zshrc |
| Cron jobs | None | No crontab entries |
| pip cache | Purged | Cleared during remediation |

### What We Did
1. Pinned `litellm>=1.80.0,<1.82.7` in both `requirements.txt` and `pyproject.toml`
2. Verified our installed version (1.80.0) is not affected
3. Purged pip cache to prevent re-installation from cached wheels
4. Conducted full security audit per official remediation steps
5. Created this security notice for future reference

### If You Suspect Compromise
If you installed versions 1.82.7 or 1.82.8:

1. **Uninstall immediately:**
   ```bash
   pip uninstall litellm
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

3. **Rotate credentials:**
   - SSH keys
   - Cloud provider credentials (AWS, GCP, Azure)
   - API keys
   - Database passwords
   - Any secrets stored in common locations

4. **Check for persistence:**
   - Review `~/.bashrc`, `~/.zshrc` for suspicious entries
   - Check cron jobs
   - Review installed site-packages for unknown `.pth` files

5. **Reinstall from safe version:**
   ```bash
   pip install "litellm>=1.80.0,<1.82.7"
   ```

### References
- [FutureSearch Analysis](https://futuresearch.ai/blog/litellm-pypi-supply-chain-attack/)
- [BerriAI Official Statement](https://litellm.ai) (check for updates)

## Reporting Security Issues

Please report security vulnerabilities by opening an issue on GitHub with the `[SECURITY]` tag.
