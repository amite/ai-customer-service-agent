# Linter Import Resolution Issue

## Problem
The linter (basedpyright) was reporting import errors for packages that were correctly installed in the virtual environment:

```
[ERROR] Line 8:8 - Import "streamlit" could not be resolved
[ERROR] Line 9:6 - Import "langchain_core.messages" could not be resolved
```

## Root Cause
The linter was not configured to recognize the virtual environment created by `uv`. The packages were correctly installed at `.venv/lib/python3.12/site-packages`, but the linter couldn't find them because:
1. No `pyrightconfig.json` existed to point to the virtual environment
2. No VS Code/Cursor settings configured the Python interpreter path

## Solution
Created two configuration files to properly configure the linter:

### 1. `pyrightconfig.json`
Created a Pyright configuration file with:
- Virtual environment path configuration (`venvPath` and `venv`)
- Correct Python version (3.12)
- Include/exclude paths for project directories
- Type checking mode set to "basic"
- Platform specification (Linux)

### 2. `.vscode/settings.json`
Created VS Code/Cursor settings with:
- Python interpreter path pointing to `.venv/bin/python`
- Extra paths for package resolution pointing to site-packages
- Type checking mode configuration

## Files Created/Modified

### Created:
- `pyrightconfig.json` - Pyright linter configuration
- `.vscode/settings.json` - Editor/IDE Python settings

### Modified:
- `streamlit_app/app.py` - Moved `langchain_core.messages` import to top of file (from previous fix)
- `pyproject.toml` - Added `langchain-core>=0.1.0` dependency (from previous fix)

## Verification
- ✅ All linter errors resolved
- ✅ Imports correctly recognized by the linter
- ✅ Virtual environment properly configured for type checking

## Technical Details
- **Virtual Environment**: `.venv` (created by `uv`)
- **Python Version**: 3.12
- **Package Location**: `.venv/lib/python3.12/site-packages`
- **Linter**: basedpyright (Pyright-based)

## Related Issues
This issue was discovered while fixing the `langchain_core.messages` import error. The package was correctly installed, but the linter couldn't resolve it due to missing configuration.
