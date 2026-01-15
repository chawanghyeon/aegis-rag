from decimal import Decimal

from aegis.domain.entities.tenant import Tenant
from aegis.domain.value_objects.api_key import ApiKey
from aegis.domain.value_objects.cost import Cost
from aegis.domain.value_objects.tenant_id import TenantId


def test_tenant_creation_generates_valid_properties():
    name = "Test Tenant"
    monthly_quota = Cost(Decimal("100.0"))

    tenant, api_key = Tenant.create(name=name, monthly_quota=monthly_quota)

    assert isinstance(tenant, Tenant)
    assert isinstance(api_key, ApiKey)
    assert isinstance(tenant.id, TenantId)
    assert tenant.name == name
    assert tenant.monthly_quota_usd == monthly_quota
    assert tenant.api_key_hash == api_key.hash()
    assert tenant.created_at is not None
    assert tenant.updated_at is not None


def test_tenant_check_quota_within_limit_returns_true():
    tenant, _ = Tenant.create("Test", Cost(Decimal("10.0")))
    current_usage = Cost(Decimal("5.0"))
    assert tenant.check_quota(current_usage) is True


def test_tenant_check_quota_exceeded_returns_false():
    tenant, _ = Tenant.create("Test", Cost(Decimal("10.0")))
    current_usage = Cost(Decimal("15.0"))
    assert tenant.check_quota(current_usage) is False
