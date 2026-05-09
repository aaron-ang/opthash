import pytest


def test_set_and_get_string_key(m):
    m["a"] = 1
    assert m["a"] == 1


def test_set_and_get_mixed_key_types(m):
    m["str"] = 1
    m[42] = 2
    m[(1, 2)] = 3
    m[frozenset([1, 2])] = 4
    assert m["str"] == 1
    assert m[42] == 2
    assert m[(1, 2)] == 3
    assert m[frozenset([1, 2])] == 4
    assert len(m) == 4


def test_getitem_missing_raises_keyerror(m):
    with pytest.raises(KeyError):
        m["missing"]


def test_delitem_missing_raises_keyerror(m):
    with pytest.raises(KeyError):
        del m["missing"]


def test_get_with_default(m):
    m["a"] = 1
    assert m.get("a") == 1
    assert m.get("missing") is None
    sentinel = object()
    assert m.get("missing", sentinel) is sentinel


def test_contains(m):
    assert "a" not in m
    m["a"] = 1
    assert "a" in m
    del m["a"]
    assert "a" not in m


def test_len_tracks_inserts_and_deletes(m):
    assert len(m) == 0
    m["a"] = 1
    m["b"] = 2
    assert len(m) == 2
    del m["a"]
    assert len(m) == 1
    m["b"] = 99  # overwrite, no growth
    assert len(m) == 1


def test_clear_resets_len(m):
    for i in range(50):
        m[i] = i
    assert len(m) == 50
    m.clear()
    assert len(m) == 0
    assert 0 not in m


@pytest.mark.parametrize("bad_key", [[1, 2, 3], {"a": 1}, {1, 2, 3}])
def test_unhashable_key_raises_typeerror(m, bad_key):
    with pytest.raises(TypeError):
        m[bad_key] = 1


def test_overwrite_returns_new_value(m):
    m["a"] = 1
    m["a"] = 2
    assert m["a"] == 2
    assert len(m) == 1


def test_capacity_property_is_readable(m):
    cap = m.capacity
    assert isinstance(cap, int)
    assert cap >= 0


def test_growth_past_initial_capacity(map_cls):
    m = map_cls(capacity=4)
    for i in range(1000):
        m[i] = i
    assert len(m) == 1000
    for i in range(1000):
        assert m[i] == i


def test_value_is_preserved_by_reference(m):
    obj = object()
    m["a"] = obj
    assert m["a"] is obj


def test_repr_contains_class_name_and_len(m):
    m["a"] = 1
    text = repr(m)
    assert "len=1" in text
    assert "HashMap" in text


def test_init_from_dict(map_cls):
    m = map_cls({"a": 1, "b": 2})
    assert len(m) == 2
    assert m["a"] == 1 and m["b"] == 2


def test_init_from_iterable_of_pairs(map_cls):
    m = map_cls([("a", 1), ("b", 2)])
    assert len(m) == 2
    assert m["a"] == 1 and m["b"] == 2


def test_init_from_other_map(map_cls):
    src = map_cls({"a": 1, "b": 2})
    m = map_cls(src)
    assert len(m) == 2
    assert m["a"] == 1 and m["b"] == 2


def test_init_with_capacity_kwarg_only(map_cls):
    m = map_cls({"a": 1}, capacity=64)
    assert m["a"] == 1
    assert m.capacity >= 64
