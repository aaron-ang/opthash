from collections.abc import ItemsView, KeysView, MutableMapping, ValuesView

from .opthash import (
    ElasticHashMap,
    FunnelHashMap,
    elastic_items,
    elastic_keys,
    elastic_values,
    funnel_items,
    funnel_keys,
    funnel_values,
)

MutableMapping.register(ElasticHashMap)
MutableMapping.register(FunnelHashMap)
KeysView.register(elastic_keys)
KeysView.register(funnel_keys)
ValuesView.register(elastic_values)
ValuesView.register(funnel_values)
ItemsView.register(elastic_items)
ItemsView.register(funnel_items)

__all__ = [
    "ElasticHashMap",
    "FunnelHashMap",
]
