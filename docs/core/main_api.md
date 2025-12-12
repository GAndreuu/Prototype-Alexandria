# 🚀 Alexandria API Documentation

**Module**: `main.py`
**Type**: FastAPI Application
**Version**: 12.0

---

## Overview
The entry point for the Alexandria Cognitive System. It exposes a REST API to interact with the Pre-Structural Field and trigger conceptual activation.

## Endpoints

### `GET /`
Returns system status.
- **Response**:
  ```json
  {
    "system": "Alexandria",
    "status": "online",
    "field_initialized": boolean
  }
  ```

### `GET /health`
Health check endpoint.
- **Response**: `{"status": "healthy"}` or `{"status": "degraded", "reason": "..."}`

### `POST /trigger`
Triggers a concept in the semantic manifold.

- **Request Body** (`TriggerRequest`):
  ```json
  {
    "embedding": [float, ...],
    "intensity": float (default: 1.0)
  }
  ```

- **Response**:
  ```json
  {
    "status": "triggered",
    "attractors": int,
    "free_energy": float
  }
  ```

## Configuration
- Initializes `PreStructuralField` on startup.
- Uses `config.settings` for manifold dimensions.

## Error Handling
- Returns `503 Service Unavailable` if the field is not initialized.
- Returns `500 Internal Server Error` for trigger failures.
