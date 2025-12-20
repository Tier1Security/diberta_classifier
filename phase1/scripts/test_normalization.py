"""
Quick script to validate `normalize_payload` behavior.

Run with:
    python3 scripts/test_normalization.py
"""
from src.normalization import normalize_payload


def test_string_normalization():
    inp = "   HeLLo   WORLD   "
    expected = "hello world"
    got = normalize_payload(inp)
    assert got == expected, f"Expected {expected}, got {got}"


def test_json_normalization():
    inp = {"Text": "   SEND   Payload ", "Other": ["A B ", "MORE    SPACES"], "Flag": True}
    got = normalize_payload(inp)
    assert got["text"] == "send payload"
    assert got["other"][0] == "a b"
    assert got["other"][1] == "more spaces"
    assert got["flag"] is True


def test_nested_list():
    inp = ["A B   C", {"Nested": "  OK  "}]
    got = normalize_payload(inp)
    assert got[0] == "a b c"
    assert got[1]["nested"] == "ok"


if __name__ == "__main__":
    test_string_normalization()
    test_json_normalization()
    test_nested_list()
    print("All normalization tests passed")
