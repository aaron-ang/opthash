import pytest

import opthash


@pytest.fixture(
    params=[opthash.ElasticHashMap, opthash.FunnelHashMap],
    ids=["elastic", "funnel"],
)
def map_cls(request):
    return request.param


@pytest.fixture
def m(map_cls):
    return map_cls(capacity=16)
