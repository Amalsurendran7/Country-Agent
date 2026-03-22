"""
Entry point.

Run directly:   python main.py
Run via CLI:    uvicorn main:app --reload
Docker:         CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import uvicorn
from api.app import app  # noqa: F401 — re-exported for uvicorn string reference

from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
