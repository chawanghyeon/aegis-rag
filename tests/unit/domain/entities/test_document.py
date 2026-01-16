from datetime import datetime
from uuid import UUID

from aegis.domain.entities.document import Document
from aegis.domain.value_objects.document_version import DocumentVersion
from aegis.domain.value_objects.tenant_id import TenantId


def test_document_creation_with_required_fields():
    tenant_id = TenantId.generate()
    filename = "test_document.pdf"
    content_hash = "a" * 64  # SHA-256 hash is 64 characters

    document = Document.create(tenant_id=tenant_id, filename=filename, content_hash=content_hash)

    assert isinstance(document, Document)
    assert isinstance(document.id, UUID)
    assert document.tenant_id == tenant_id
    assert document.filename == filename
    assert document.content_hash == content_hash
    assert document.status == "pending"
    assert document.version == DocumentVersion()
    assert document.is_latest is True
    assert document.previous_version_id is None
    assert document.replaced_at is None
    assert isinstance(document.created_at, datetime)
