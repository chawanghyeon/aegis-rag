from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class TenantId:
    """테넌트 ID 값 객체 — 테넌트를 위한 불변 식별자."""

    value: UUID

    def __post_init__(self) -> None:
        """초기화 시 UUID 유효성 검사."""
        if not isinstance(self.value, UUID):
            raise TypeError(f"TenantId는 UUID여야 합니다. 받은 타입: {type(self.value)}")
