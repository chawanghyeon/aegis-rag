import hashlib

import pytest

from aegis.domain.value_objects.api_key import ApiKey


def test_api_key_validation_raises_error_when_too_short():
    with pytest.raises(ValueError, match="API Key must be at least 32 characters long"):
        ApiKey("short_key")


def test_api_key_creation_succeeds_with_valid_length():
    valid_key = "a" * 32
    api_key = ApiKey(valid_key)
    assert str(api_key) == valid_key


def test_api_key_generate_creates_valid_key():
    api_key = ApiKey.generate()
    assert isinstance(api_key, ApiKey)
    assert len(str(api_key)) >= 32


def test_api_key_generate_creates_unique_keys():
    key1 = ApiKey.generate()
    key2 = ApiKey.generate()
    assert str(key1) != str(key2)


def test_api_key_hash_returns_sha256_digest():
    raw_key = "a" * 32
    api_key = ApiKey(raw_key)
    expected_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    assert api_key.hash() == expected_hash


def test_api_key_hash_is_stable():
    api_key = ApiKey.generate()
    assert api_key.hash() == api_key.hash()
