from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal


@dataclass(frozen=True)
class Cost:
    amount: Decimal

    def __post_init__(self):
        # Convert to Decimal if not already, using object.__setattr__ because frozen=True
        if not isinstance(self.amount, Decimal):
            object.__setattr__(self, "amount", Decimal(str(self.amount)))

        # Quantize to 6 decimal places
        quantized = self.amount.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
        object.__setattr__(self, "amount", quantized)

        if self.amount < 0:
            raise ValueError("Cost amount cannot be negative")

    def __str__(self):
        return f"${self.amount:.6f}"

    def add(self, other: "Cost") -> "Cost":
        return Cost(self.amount + other.amount)

    def __add__(self, other: "Cost") -> "Cost":
        return self.add(other)
