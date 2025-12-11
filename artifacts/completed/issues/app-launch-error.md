# Streamlit App Launch Error: ModuleNotFoundError

## Issue
When starting the Streamlit app (`streamlit_app/app.py`), the following error occurred:
```
ModuleNotFoundError: No module named 'vectorstore'
```

The error occurred at line 19 in `src/agents/customer_service_agent.py` during import.

## Root Cause
The import statement used an incorrect module path:
```python
from vectorstore.vector_store import initialize_vector_store, search_similar_products, search_faqs
```

Since the project structure has `src` as the root package (as defined in `pyproject.toml` with `packages = ["src"]`), the import path must be prefixed with `src.`.

## Fix
Changed the import statement in `src/agents/customer_service_agent.py` (line 19) from:
```python
from vectorstore.vector_store import initialize_vector_store, search_similar_products, search_faqs
```

to:
```python
from src.vectorstore.vector_store import initialize_vector_store, search_similar_products, search_faqs
```

## Verification
- Import tested successfully with `uv run python`
- Debug logs confirmed "Import successful" message
- Module resolution verified: `src.vectorstore.vector_store` is found correctly

## Resolution Date
2025-12-11
