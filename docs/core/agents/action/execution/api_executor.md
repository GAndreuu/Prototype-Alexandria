# 🌐 API Executor

**Module**: `core/agents/action/execution/api_executor.py`  
**Lines of Code**: 81  
**Purpose**: Handles API call execution with security validation.

---

## 🎯 Overview

Executes HTTP API calls with validation through SecurityController.

---

## 📦 Function: `execute_api_call`

```python
def execute_api_call(
    parameters: Dict[str, Any],
    security_controller: SecurityController,
    action_id: str
) -> ActionResult
```

### Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | required | Target URL |
| `method` | str | "GET" | HTTP method (GET, POST, PUT, DELETE) |
| `headers` | dict | {} | Request headers |
| `data` | any | None | Request body for POST/PUT |
| `timeout` | int | 30 | Request timeout in seconds |

### Returns
```python
{
    "status_code": 200,
    "headers": {...},
    "response_size": 1234,
    "success": True,
    "json_response": {...}  # or text_response
}
```

---

## 🔒 Security
Validates URL against whitelist via `security_controller.validate_api_call(url)`.

---

**Last Updated**: 2025-12-13  
**Version**: 2.0  
**Status**: Active
