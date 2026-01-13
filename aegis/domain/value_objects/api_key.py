import hashlib
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class ApiKey:
    value: str

    def __post_init__(self):
        if len(self.value) < 32:
            raise ValueError("API Key must be at least 32 characters long")

    def __str__(self):
        return self.value

    @classmethod
    def generate(cls) -> "ApiKey":
        return cls(secrets.token_urlsafe(32))

    def hash(self) -> str:
        return hashlib.sha256(self.value.encode()).hexdigest()
