# 🔒 SecurityController

**Module**: `core/agents/action/security_controller.py`  
**Lines of Code**: 96  
**Purpose**: Security and audit control for Action Agent operations.

---

## 🎯 Overview

Handles API validation, rate limiting, and audit logging to ensure safe action execution.

---

## 🏗️ Class: SecurityController

```python
class SecurityController:
    def __init__(self)
```

### Attributes
- `allowed_apis`: List of whitelisted API domains (from `ALLOWED_APIS` env var)
- `rate_limits`: Dict tracking action timestamps per user/type
- `audit_log`: List of action log entries
- `blocked_domains`: Set of blocked domains

---

## 🔄 Methods

### `validate_api_call(url) → bool`
Validates if API URL is allowed:
- Checks against blocked domains
- Verifies whitelist if configured
- Denies by default if no whitelist

### `check_rate_limit(action_type, user_id) → bool`
Rate limiting per action type:
- **Window**: 5 minutes
- **Limit**: 10 actions per window
- Returns `False` if limit exceeded

### `log_action(action, details)`
Records action to audit log:
- Keeps last 1000 entries
- Includes timestamp, action_id, type, status, details

### `get_audit_log(limit) → List[Dict]`
Returns recent audit log entries.

---

## ⚙️ Configuration

Environment variable:
```bash
ALLOWED_APIS="api.arxiv.org,api.semanticscholar.org"
```

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
