import pytest

import opthash


def test_elastic_options_defaults():
    opts = opthash.ElasticOptions()
    assert opts.capacity == 0
    assert 0.0 < opts.reserve_fraction < 1.0
    assert opts.probe_scale > 0.0


def test_elastic_options_custom_kwargs():
    opts = opthash.ElasticOptions(
        capacity=1024,
        reserve_fraction=0.1,
        probe_scale=8.0,
    )
    assert opts.capacity == 1024
    assert opts.reserve_fraction == pytest.approx(0.1)
    assert opts.probe_scale == pytest.approx(8.0)


def test_elastic_options_reject_invalid_reserve_fraction():
    with pytest.raises(ValueError):
        opthash.ElasticOptions(reserve_fraction=2.0)
    with pytest.raises(ValueError):
        opthash.ElasticOptions(reserve_fraction=0.0)
    with pytest.raises(ValueError):
        opthash.ElasticOptions(reserve_fraction=-0.1)


def test_elastic_options_reject_invalid_probe_scale():
    with pytest.raises(ValueError):
        opthash.ElasticOptions(probe_scale=0.0)
    with pytest.raises(ValueError):
        opthash.ElasticOptions(probe_scale=-1.0)


def test_elastic_with_options_classmethod():
    opts = opthash.ElasticOptions(capacity=64, reserve_fraction=0.1, probe_scale=8.0)
    m = opthash.ElasticHashMap.with_options(opts)
    for i in range(50):
        m[i] = i * 2
    for i in range(50):
        assert m[i] == i * 2


def test_funnel_options_defaults():
    opts = opthash.FunnelOptions()
    assert opts.capacity == 0
    assert 0.0 < opts.reserve_fraction < 1.0
    assert opts.primary_probe_limit is None


def test_funnel_options_custom_kwargs():
    opts = opthash.FunnelOptions(
        capacity=2048,
        reserve_fraction=0.1,
        primary_probe_limit=16,
    )
    assert opts.capacity == 2048
    assert opts.reserve_fraction == pytest.approx(0.1)
    assert opts.primary_probe_limit == 16


def test_funnel_options_reject_invalid_reserve_fraction():
    with pytest.raises(ValueError):
        opthash.FunnelOptions(reserve_fraction=1.0)
    with pytest.raises(ValueError):
        opthash.FunnelOptions(reserve_fraction=0.5)  # above 1/8 cap
    with pytest.raises(ValueError):
        opthash.FunnelOptions(reserve_fraction=0.0)
    with pytest.raises(ValueError):
        opthash.FunnelOptions(reserve_fraction=-0.1)


def test_funnel_options_accept_max_reserve_fraction():
    opts = opthash.FunnelOptions(reserve_fraction=0.125)
    assert opts.reserve_fraction == pytest.approx(0.125)


def test_funnel_options_reject_zero_probe_limit():
    with pytest.raises(ValueError):
        opthash.FunnelOptions(primary_probe_limit=0)


def test_funnel_with_options_classmethod():
    opts = opthash.FunnelOptions(capacity=128, reserve_fraction=0.1)
    m = opthash.FunnelHashMap.with_options(opts)
    for i in range(100):
        m[f"k{i}"] = i
    for i in range(100):
        assert m[f"k{i}"] == i
