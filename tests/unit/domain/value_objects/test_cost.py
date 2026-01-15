from decimal import Decimal

import pytest

from aegis.domain.value_objects.cost import Cost


def test_cost_creation_and_formatting():
    cost = Cost(Decimal("10.123456"))
    assert str(cost) == "$10.123456"


def test_cost_creation_from_float():
    cost = Cost(10.5)
    assert cost.amount == Decimal("10.500000")
    assert str(cost) == "$10.500000"


def test_cost_creation_from_int():
    cost = Cost(10)
    assert cost.amount == Decimal("10.000000")
    assert str(cost) == "$10.000000"


def test_cost_raises_error_for_negative_value():
    with pytest.raises(ValueError, match="Cost amount cannot be negative"):
        Cost(Decimal("-1.0"))


def test_cost_addition_and_immutability():
    c1 = Cost(10)
    c2 = Cost(5.5)
    c3 = c1.add(c2)

    assert c3.amount == Decimal("15.500000")
    assert str(c3) == "$15.500000"
    assert c1.amount == Decimal("10.000000")  # Check immutability

    # Test __add__ alias
    c4 = c1 + c2
    assert c4 == c3
