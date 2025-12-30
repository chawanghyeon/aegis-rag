import uuid
from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest


def test_tenant_id_creation_with_valid_uuid():
    """테스트: TenantId 생성 시 유효한 UUID 검증"""
    # Given: 유효한 UUID
    valid_uuid = uuid.uuid4()

    # When: TenantId를 생성하면
    from aegis.domain.value_objects.tenant_id import TenantId

    tenant_id = TenantId(valid_uuid)

    # Then: 성공적으로 생성되고 값이 일치한다
    assert tenant_id.value == valid_uuid
    assert isinstance(tenant_id.value, UUID)


def test_tenant_id_is_immutable():
    """테스트: TenantId는 불변 (immutable)"""
    # Given: TenantId 객체
    from aegis.domain.value_objects.tenant_id import TenantId

    tenant_id = TenantId(uuid.uuid4())

    # When/Then: value 속성을 변경하려고 하면 에러가 발생한다
    with pytest.raises((FrozenInstanceError, AttributeError)):
        tenant_id.value = uuid.uuid4()  # type: ignore
