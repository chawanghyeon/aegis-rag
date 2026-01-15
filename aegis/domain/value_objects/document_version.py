from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentVersion:
    value: int = 1

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Version must be greater than 0")

    def __str__(self):
        return str(self.value)

    def increment(self) -> "DocumentVersion":
        return DocumentVersion(self.value + 1)
