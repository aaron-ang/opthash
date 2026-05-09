"""Microbench breaking down per-op overhead in opthash Python bindings.

Decomposes the cost of one `m[k]` call by comparing primitives:

    raw loop  -> hash(k)  -> dict[k]  -> opthash __contains__  -> opthash __getitem__

Δ between adjacent rows is the cost of one extra primitive on the path.
"""

import time
import opthash

N = 10_000
ITERS = 50  # repeat the inner N-loop this many times for stable timing
KEYS_STR = [f"key_{i}" for i in range(N)]
KEYS_INT = list(range(N))


def bench(name, fn):
    # warmup
    for _ in range(3):
        fn()
    samples = []
    for _ in range(ITERS):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    ns_per_op = median / N
    print(f"{name:<45} {ns_per_op:>7.1f} ns/op   (median {median/1e3:>7.1f} µs / {N} ops)")
    return ns_per_op


def make_loaded(cls, keys):
    m = cls(capacity=N)
    for k in keys:
        m[k] = 0
    return m


def main():
    print(f"=== str keys, N={N} ===")
    keys = KEYS_STR
    d = dict(zip(keys, [0] * N))
    e = make_loaded(opthash.ElasticHashMap, keys)
    f = make_loaded(opthash.FunnelHashMap, keys)

    raw = bench("loop only (pass)", lambda: [None for _ in keys])
    h = bench("hash(k)", lambda: [hash(k) for k in keys])
    bench("dict[k]", lambda: [d[k] for k in keys])

    # contains skips value clone_ref → isolates HashedAny + map probe
    bench("elastic k in m", lambda: [k in e for k in keys])
    bench("elastic m[k]", lambda: [e[k] for k in keys])
    bench("elastic m.get(k)", lambda: [e.get(k) for k in keys])

    bench("funnel k in m", lambda: [k in f for k in keys])
    bench("funnel m[k]", lambda: [f[k] for k in keys])
    bench("funnel m.get(k)", lambda: [f.get(k) for k in keys])

    print(f"\n=== int keys, N={N} ===")
    keys = KEYS_INT
    d = dict(zip(keys, [0] * N))
    e = make_loaded(opthash.ElasticHashMap, keys)
    f = make_loaded(opthash.FunnelHashMap, keys)

    bench("dict[k]", lambda: [d[k] for k in keys])
    bench("elastic m[k]", lambda: [e[k] for k in keys])
    bench("funnel m[k]", lambda: [f[k] for k in keys])

    print(f"\n=== insert breakdown, str keys, N={N} ===")
    bench(
        "dict insert",
        lambda: (lambda dd: [dd.__setitem__(k, 0) for k in KEYS_STR])(dict()),
    )
    bench(
        "elastic insert (capacity=N)",
        lambda: (lambda mm: [mm.__setitem__(k, 0) for k in KEYS_STR])(
            opthash.ElasticHashMap(capacity=N)
        ),
    )
    bench(
        "funnel insert (capacity=N)",
        lambda: (lambda mm: [mm.__setitem__(k, 0) for k in KEYS_STR])(
            opthash.FunnelHashMap(capacity=N)
        ),
    )

    print(f"\n=== get_miss str keys, N={N} ===")
    miss = [f"miss_{i}" for i in range(N)]
    bench("dict.get(miss)", lambda: [d.get(k) for k in miss])
    bench("elastic.get(miss)", lambda: [e.get(k) for k in miss])
    bench("funnel.get(miss)", lambda: [f.get(k) for k in miss])


if __name__ == "__main__":
    main()
