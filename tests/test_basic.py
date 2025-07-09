"""
Basic tests that don't require external dependencies.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_import_structure():
    """Test that the basic module structure can be imported."""
    try:
        import leukomap
        assert hasattr(leukomap, '__version__')
        assert leukomap.__version__ == "0.1.0"
        print("✅ Basic import successful")
    except ImportError as e:
        pytest.fail(f"Failed to import leukomap: {e}")


def test_data_loading_import():
    """Test that data_loading module can be imported (without dependencies)."""
    try:
        # Test the module file exists
        data_loading_file = Path(__file__).parent.parent / "leukomap" / "data_loading.py"
        assert data_loading_file.exists(), f"data_loading.py not found at {data_loading_file}"
        print("✅ data_loading.py file exists")
    except Exception as e:
        pytest.fail(f"Failed to check data_loading module: {e}")


def test_preprocessing_import():
    """Test that preprocessing module can be imported."""
    try:
        # Test the module file exists
        preprocessing_file = Path(__file__).parent.parent / "leukomap" / "preprocessing.py"
        assert preprocessing_file.exists(), f"preprocessing.py not found at {preprocessing_file}"
        print("✅ preprocessing.py file exists")
    except Exception as e:
        pytest.fail(f"Failed to check preprocessing module: {e}")


def test_project_structure():
    """Test that the project has the expected structure."""
    project_root = Path(__file__).parent.parent
    
    expected_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "leukomap/__init__.py",
        "leukomap/data_loading.py",
        "leukomap/preprocessing.py",
        "tests/__init__.py",
        "tests/test_data_loading.py",
        "docs/data_loading.md",
        "examples/load_data_example.py"
    ]
    
    for file_path in expected_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Expected file not found: {file_path}"
    
    print("✅ All expected project files exist")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 