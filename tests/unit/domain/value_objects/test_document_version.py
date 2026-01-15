import pytest

from aegis.domain.value_objects.document_version import DocumentVersion


def test_document_version_default_creation():
    version = DocumentVersion()
    assert version.value == 1


def test_document_version_raises_error_for_invalid_value():
    with pytest.raises(ValueError, match="Version must be greater than 0"):
        DocumentVersion(0)
    with pytest.raises(ValueError, match="Version must be greater than 0"):
        DocumentVersion(-1)


def test_document_version_increment_increases_value():
    v1 = DocumentVersion(1)
    v2 = v1.increment()
    assert v2.value == 2
    assert v1.value == 1  # Verify immutability
