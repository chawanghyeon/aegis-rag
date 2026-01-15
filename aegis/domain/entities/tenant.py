from dataclasses import dataclass
from datetime import UTC, datetime

from aegis.domain.value_objects.api_key import ApiKey
from aegis.domain.value_objects.cost import Cost
from aegis.domain.value_objects.tenant_id import TenantId


@dataclass
class Tenant:
    id: TenantId
    name: str
    api_key_hash: str
    monthly_quota_usd: Cost
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(cls, name: str, monthly_quota: Cost) -> tuple["Tenant", ApiKey]:
        tenant_id = TenantId.generate()
        api_key = ApiKey.generate()
        now = datetime.now(UTC)

        tenant = cls(
            id=tenant_id,
            name=name,
            api_key_hash=api_key.hash(),
            monthly_quota_usd=monthly_quota,
            created_at=now,
            updated_at=now,
        )
        return tenant, api_key

    def check_quota(self, current_usage: Cost) -> bool:
        return current_usage.amount <= self.monthly_quota_usd.amount
