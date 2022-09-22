import pytest
import cstrees


def test_project_defines_author_and_version():
    assert hasattr(cstrees, '__author__')
    assert hasattr(cstrees, '__version__')
