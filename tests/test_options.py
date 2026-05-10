import pytest

import opthash


def test_elastic_with_options_defaults():
    m = opthash.ElasticHashMap.with_options()
    assert len(m) == 0
    m[1] = 2
    assert m[1] == 2


def test_elastic_with_options_custom_kwargs():
    m = opthash.ElasticHashMap.with_options(
        capacity=64, reserve_fraction=0.1, probe_scale=8.0
    )
    for i in range(50):
        m[i] = i * 2
    for i in range(50):
        assert m[i] == i * 2


def test_elastic_with_options_reject_invalid_reserve_fraction():
    with pytest.raises(ValueError):
        opthash.ElasticHashMap.with_options(reserve_fraction=2.0)
    with pytest.raises(ValueError):
        opthash.ElasticHashMap.with_options(reserve_fraction=0.0)
    with pytest.raises(ValueError):
        opthash.ElasticHashMap.with_options(reserve_fraction=-0.1)


def test_elastic_with_options_reject_invalid_probe_scale():
    with pytest.raises(ValueError):
        opthash.ElasticHashMap.with_options(probe_scale=0.0)
    with pytest.raises(ValueError):
        opthash.ElasticHashMap.with_options(probe_scale=-1.0)


def test_funnel_with_options_defaults():
    m = opthash.FunnelHashMap.with_options()
    assert len(m) == 0
    m["a"] = 1
    assert m["a"] == 1


def test_funnel_with_options_custom_kwargs():
    m = opthash.FunnelHashMap.with_options(
        capacity=128, reserve_fraction=0.1, primary_probe_limit=16
    )
    for i in range(100):
        m[f"k{i}"] = i
    for i in range(100):
        assert m[f"k{i}"] == i


def test_funnel_with_options_reject_invalid_reserve_fraction():
    with pytest.raises(ValueError):
        opthash.FunnelHashMap.with_options(reserve_fraction=1.0)
    with pytest.raises(ValueError):
        opthash.FunnelHashMap.with_options(reserve_fraction=0.5)  # above 1/8 cap
    with pytest.raises(ValueError):
        opthash.FunnelHashMap.with_options(reserve_fraction=0.0)
    with pytest.raises(ValueError):
        opthash.FunnelHashMap.with_options(reserve_fraction=-0.1)


def test_funnel_with_options_accept_max_reserve_fraction():
    m = opthash.FunnelHashMap.with_options(reserve_fraction=0.125)
    m["x"] = 1
    assert m["x"] == 1


def test_funnel_with_options_reject_zero_probe_limit():
    with pytest.raises(ValueError):
        opthash.FunnelHashMap.with_options(primary_probe_limit=0)
