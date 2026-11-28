from typing import Dict, Any

DEFAULT_JSON: Dict[str, Any] = {
    "request_id": "req-20240318-001",
    "timestamp": "2024-03-18T10:32:11Z",
    "environment": "production",
    "service": {
        "name": "internal-query-service",
        "version": "v1.4.2",
        "region": "us-east",
    },
    "client": {
        "id": "anonymous",
        "type": "internal",
        "permissions": ["read", "query", "analyze"],
    },
    "metadata": {
        "source": "upstream_system",
        "correlation_id": "corr-88921",
        "retry_count": 0,
        "flags": {"dry_run": False, "debug": False, "experimental": False},
    },
    "request": {
        "type": "query",
        "priority": "normal",
        "parameters": {
            "language": "en",
            "format": "text",
            "max_results": 10,
            "include_metadata": True,
        },
        "filters": {
            "date_range": {"start": "2024-01-01", "end": "2024-03-18"},
            "categories": ["general", "internal", "reference"],
        },
    },
    "context": {
        "summary": "Standard internal workflow request.",
        "notes": "Additional contextual information may be provided.",
        "history": [
            {
                "event": "created",
                "actor": "system",
                "timestamp": "2024-03-18T10:32:10Z",
            },
            {
                "event": "validated",
                "actor": "system",
                "timestamp": "2024-03-18T10:32:11Z",
            },
        ],
    },
    "processing": {
        "status": "pending",
        "steps": [
            {
                "name": "input_validation",
                "status": "completed",
                "duration_ms": 12,
            },
            {
                "name": "context_enrichment",
                "status": "completed",
                "duration_ms": 34,
            },
            {"name": "execution", "status": "pending", "duration_ms": None},
        ],
    },
    "response": {
        "format": "text",
        "delivery": {"method": "synchronous", "timeout_ms": 3000},
        "placeholders": {"content": "Pending execution.", "confidence": None},
    },
    "audit": {
        "logged": True,
        "log_level": "INFO",
        "retention_days": 30,
        "tags": ["internal", "automated", "non_user_facing"],
    },
}
