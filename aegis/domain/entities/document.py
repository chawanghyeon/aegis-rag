from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from aegis.domain.value_objects.document_version import DocumentVersion
from aegis.domain.value_objects.tenant_id import TenantId


@dataclass
class Document:
    id: UUID
    tenant_id: TenantId
    filename: str
    content_hash: str
    status: str
    version: DocumentVersion
    is_latest: bool
    previous_version_id: UUID | None
    replaced_at: datetime | None
    created_at: datetime

    @classmethod
    def create(cls, tenant_id: TenantId, filename: str, content_hash: str) -> "Document":
        return cls(
            id=uuid4(),
            tenant_id=tenant_id,
            filename=filename,
            content_hash=content_hash,
            status="pending",
            version=DocumentVersion(),
            is_latest=True,
            previous_version_id=None,
            replaced_at=None,
            created_at=datetime.now(UTC),
        )
