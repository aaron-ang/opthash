"""Coverage for dict-parity API: iteration, bulk ops, equality, merge."""

import pytest


def _populate(m, n=10):
    for i in range(n):
        m[f"k{i}"] = i
    return m


def test_iter_yields_all_keys(m):
    _populate(m, 10)
    seen = set()
    for k in m:
        seen.add(k)
    assert seen == {f"k{i}" for i in range(10)}


def test_iter_empty_map(m):
    assert list(m) == []


def test_keys_returns_list_of_keys(m):
    _populate(m, 5)
    keys = m.keys()
    assert isinstance(keys, list)
    assert sorted(keys) == [f"k{i}" for i in range(5)]


def test_values_returns_list_of_values(m):
    _populate(m, 5)
    vals = m.values()
    assert isinstance(vals, list)
    assert sorted(vals) == list(range(5))


def test_items_returns_list_of_pairs(m):
    _populate(m, 5)
    items = m.items()
    assert isinstance(items, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in items)
    assert sorted(items) == [(f"k{i}", i) for i in range(5)]


def test_dict_comprehension_works(m):
    _populate(m, 5)
    d = {k: v for k, v in m.items()}
    assert d == {f"k{i}": i for i in range(5)}


def test_list_iter_snapshot(m):
    _populate(m, 3)
    snapshot = list(m)
    m["new"] = 99
    assert "new" not in snapshot  # snapshot was taken before mutation


def test_update_from_dict(m):
    _populate(m, 3)
    m.update({"k0": 100, "new": 42})
    assert m["k0"] == 100  # overwrite
    assert m["new"] == 42  # added
    assert len(m) == 4


def test_update_from_iterable_of_pairs(m):
    m.update([("a", 1), ("b", 2), ("c", 3)])
    assert m["a"] == 1
    assert m["b"] == 2
    assert m["c"] == 3
    assert len(m) == 3


def test_update_from_iterable_rejects_non_pair(m):
    with pytest.raises(ValueError):
        m.update([("a", 1), ("b", 2, 3)])


def test_pop_returns_and_removes(m):
    _populate(m, 3)
    val = m.pop("k1")
    assert val == 1
    assert "k1" not in m
    assert len(m) == 2


def test_pop_missing_returns_default(m):
    sentinel = object()
    assert m.pop("missing", sentinel) is sentinel


def test_pop_missing_no_default_raises(m):
    with pytest.raises(KeyError):
        m.pop("missing")


def test_popitem_returns_and_removes_some_item(m):
    _populate(m, 3)
    k, v = m.popitem()
    assert isinstance(k, str)
    assert isinstance(v, int)
    assert k.startswith("k") and v == int(k[1:])
    assert k not in m
    assert len(m) == 2


def test_popitem_empty_raises(m):
    with pytest.raises(KeyError):
        m.popitem()


def test_setdefault_inserts_when_missing(m):
    val = m.setdefault("a", 42)
    assert val == 42
    assert m["a"] == 42


def test_setdefault_returns_existing(m):
    m["a"] = 1
    val = m.setdefault("a", 999)
    assert val == 1
    assert m["a"] == 1  # unchanged


def test_setdefault_default_is_none(m):
    val = m.setdefault("a")
    assert val is None
    assert m["a"] is None


def test_copy_independent(m):
    _populate(m, 3)
    c = m.copy()
    assert len(c) == 3
    c["new"] = 99
    assert "new" not in m  # mutating copy doesn't affect original
    m["other"] = 100
    assert "other" not in c  # mutating original doesn't affect copy


def test_eq_with_same_map(m, map_cls):
    _populate(m, 5)
    other = map_cls()
    _populate(other, 5)
    assert m == other


def test_eq_with_dict(m):
    _populate(m, 3)
    assert m == {"k0": 0, "k1": 1, "k2": 2}


def test_eq_size_mismatch(m):
    _populate(m, 3)
    assert m != {"k0": 0, "k1": 1}
    assert m != {"k0": 0, "k1": 1, "k2": 2, "k3": 3}


def test_eq_value_mismatch(m):
    _populate(m, 3)
    assert m != {"k0": 0, "k1": 999, "k2": 2}


def test_or_returns_merged_map(m, map_cls):
    _populate(m, 3)
    new = m | {"k0": 100, "extra": 7}
    assert isinstance(new, map_cls)
    assert new["k0"] == 100  # other overrides
    assert new["k1"] == 1
    assert new["extra"] == 7
    assert "extra" not in m  # original untouched


def test_or_with_another_map(m, map_cls):
    _populate(m, 2)
    other = map_cls()
    other["x"] = 99
    new = m | other
    assert new["k0"] == 0
    assert new["x"] == 99
