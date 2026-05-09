import pytest
from hypothesis import given, settings, strategies as st

import opthash


KEYS = st.one_of(
    st.integers(-100, 100),
    st.text(alphabet="abcdefgh", min_size=0, max_size=4),
    st.tuples(st.integers(0, 4), st.integers(0, 4)),
)

OPS = st.lists(
    st.tuples(
        st.sampled_from(["set", "del", "get", "in"]),
        KEYS,
        st.integers(),
    ),
    min_size=0,
    max_size=500,
)


@pytest.mark.parametrize(
    "cls",
    [opthash.ElasticHashMap, opthash.FunnelHashMap],
    ids=["elastic", "funnel"],
)
@given(ops=OPS)
@settings(max_examples=100, deadline=None)
def test_parity_with_dict(cls, ops):
    ref: dict = {}
    m = cls(capacity=8)

    for op, k, v in ops:
        if op == "set":
            ref[k] = v
            m[k] = v
        elif op == "del":
            ref_err = m_err = None
            try:
                del ref[k]
            except KeyError:
                ref_err = KeyError
            try:
                del m[k]
            except KeyError:
                m_err = KeyError
            assert ref_err == m_err, f"divergent KeyError on del {k!r}"
        elif op == "get":
            assert m.get(k) == ref.get(k), f"divergent get on {k!r}"
        elif op == "in":
            assert (k in m) == (k in ref), f"divergent contains on {k!r}"

        assert len(m) == len(ref), f"len divergence after {op} {k!r}"

    for k, v in ref.items():
        assert k in m
        assert m[k] == v
