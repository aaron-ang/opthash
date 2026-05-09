import random

import pytest

import opthash


N = 10_000
SEED = 0


# All three factories grow from zero. dict() doesn't accept a capacity hint, so
# pre-allocating opthash maps would give them an unfair head-start.
def _factory_dict(_n: int):
    return dict()


def _factory_elastic(_n: int):
    return opthash.ElasticHashMap()


def _factory_funnel(_n: int):
    return opthash.FunnelHashMap()


IMPLS = [
    pytest.param(_factory_dict, id="dict"),
    pytest.param(_factory_elastic, id="elastic"),
    pytest.param(_factory_funnel, id="funnel"),
]


@pytest.fixture(scope="module")
def keys() -> list[str]:
    return [f"key_{i}" for i in range(N)]


@pytest.fixture(scope="module")
def miss_keys() -> list[str]:
    return [f"miss_{i}" for i in range(N)]


@pytest.fixture(scope="module")
def mixed_indices() -> list[int]:
    rng = random.Random(SEED)
    return [rng.randrange(N) for _ in range(N)]


@pytest.mark.benchmark(group="insert")
@pytest.mark.parametrize("factory", IMPLS)
def test_insert(benchmark, factory, keys):
    def run():
        m = factory(N)
        for k in keys:
            m[k] = 0

    benchmark(run)


@pytest.mark.benchmark(group="get_hit")
@pytest.mark.parametrize("factory", IMPLS)
def test_get_hit(benchmark, factory, keys):
    m = factory(N)
    for k in keys:
        m[k] = 0

    def run():
        for k in keys:
            _ = m[k]

    benchmark(run)


@pytest.mark.benchmark(group="get_miss")
@pytest.mark.parametrize("factory", IMPLS)
def test_get_miss(benchmark, factory, keys, miss_keys):
    m = factory(N)
    for k in keys:
        m[k] = 0

    def run():
        for k in miss_keys:
            _ = m.get(k)

    benchmark(run)


@pytest.mark.benchmark(group="mixed")
@pytest.mark.parametrize("factory", IMPLS)
def test_mixed(benchmark, factory, keys, mixed_indices):
    m = factory(N)
    for k in keys:
        m[k] = 0

    def run():
        for i in mixed_indices:
            k = keys[i]
            if i & 1:
                m[k] = i
            else:
                _ = m[k]

    benchmark(run)


@pytest.mark.benchmark(group="delete")
@pytest.mark.parametrize("factory", IMPLS)
def test_delete(benchmark, factory, keys):
    def run():
        m = factory(N)
        for k in keys:
            m[k] = 0
        for k in keys:
            del m[k]

    benchmark(run)
