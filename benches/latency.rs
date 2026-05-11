mod common;

use std::fs;
use std::hint::black_box;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use common::{
    GOLDEN_RATIO_U64, build_elastic_map, build_funnel_map, build_hashbrown_map, build_std_map,
    make_pairs,
};
use hdrhistogram::Histogram;

const MAPS: &[&str] = &["std", "hashbrown", "elastic", "funnel"];
const SIZE: usize = 10_000_000;
const OP: &str = "get-hit";
const SAMPLES: usize = 1_000_000;
const WARMUP: usize = 10_000;

fn elapsed_ns(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).expect("elapsed fits in u64")
}

fn measure_clock_overhead_ns() -> u64 {
    let n = 10_000u64;
    let t0 = Instant::now();
    for _ in 0..n {
        black_box(Instant::now());
    }
    elapsed_ns(t0) / n
}

fn new_hist() -> Histogram<u64> {
    Histogram::<u64>::new_with_bounds(1, 1_000_000_000, 3).expect("valid hdr bounds")
}

fn scatter(i: usize, n: usize) -> usize {
    #[allow(clippy::cast_possible_truncation)]
    let mixed = (i as u64).wrapping_mul(GOLDEN_RATIO_U64) as usize;
    mixed % n
}

fn measure<F, R>(samples: usize, warmup: usize, mut op: F) -> Histogram<u64>
where
    F: FnMut(usize) -> R,
{
    for i in 0..warmup {
        black_box(op(i));
    }
    let mut h = new_hist();
    for i in 0..samples {
        let t0 = Instant::now();
        let r = op(i);
        let dt = elapsed_ns(t0);
        black_box(r);
        h.record(dt.max(1)).unwrap();
    }
    h
}

fn run_get_hit(map: &str, size: usize, samples: usize, warmup: usize) -> Histogram<u64> {
    let pairs = make_pairs(size);
    let keys: Vec<u64> = pairs.iter().map(|&(k, _)| k).collect();
    let n = keys.len();
    match map {
        "std" => {
            let m = build_std_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        "hashbrown" => {
            let m = build_hashbrown_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        "elastic" => {
            let m = build_elastic_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        "funnel" => {
            let m = build_funnel_map(&pairs);
            measure(samples, warmup, |i| {
                m.get(black_box(&keys[scatter(i, n)])).copied()
            })
        }
        _ => unreachable!(),
    }
}

fn write_json(
    map: &str,
    h: &Histogram<u64>,
    overhead: u64,
    samples: usize,
) -> std::io::Result<PathBuf> {
    let dir = PathBuf::from(format!("target/latency/{map}/{SIZE}"));
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{OP}.json"));
    let mut f = fs::File::create(&path)?;

    let p50 = h.value_at_quantile(0.50);
    let p90 = h.value_at_quantile(0.90);
    let p99 = h.value_at_quantile(0.99);
    let p999 = h.value_at_quantile(0.999);
    let p9999 = h.value_at_quantile(0.9999);
    let p99999 = h.value_at_quantile(0.99999);
    let max = h.max();
    let mean = h.mean();
    let cap = p99999.saturating_mul(2).max(p9999.saturating_mul(4));

    writeln!(f, "{{")?;
    writeln!(f, "  \"map\": \"{map}\",")?;
    writeln!(f, "  \"size\": {SIZE},")?;
    writeln!(f, "  \"op\": \"{OP}\",")?;
    writeln!(f, "  \"samples\": {samples},")?;
    writeln!(f, "  \"clock_overhead_ns\": {overhead},")?;
    writeln!(f, "  \"percentiles\": {{")?;
    writeln!(f, "    \"p50\": {p50},")?;
    writeln!(f, "    \"p90\": {p90},")?;
    writeln!(f, "    \"p99\": {p99},")?;
    writeln!(f, "    \"p999\": {p999},")?;
    writeln!(f, "    \"p9999\": {p9999},")?;
    writeln!(f, "    \"p99999\": {p99999},")?;
    writeln!(f, "    \"max\": {max},")?;
    writeln!(f, "    \"mean\": {mean:.3}")?;
    writeln!(f, "  }},")?;
    write!(f, "  \"histogram\": [")?;
    let mut first = true;
    for v in h.iter_recorded() {
        let hi = v.value_iterated_to();
        if hi > cap {
            break;
        }
        let count = v.count_since_last_iteration();
        if count == 0 {
            continue;
        }
        let lo = h.lowest_equivalent(hi);
        if first {
            writeln!(f)?;
        } else {
            writeln!(f, ",")?;
        }
        first = false;
        write!(
            f,
            "    {{\"ns_low\": {lo}, \"ns_high\": {hi}, \"count\": {count}}}"
        )?;
    }
    if !first {
        writeln!(f)?;
        write!(f, "  ")?;
    }
    writeln!(f, "]")?;
    writeln!(f, "}}")?;
    Ok(path)
}

fn main() {
    let overhead = measure_clock_overhead_ns();
    eprintln!("clock_overhead_ns ≈ {overhead}");

    for &m in MAPS {
        eprint!("running map={m} size={SIZE} op={OP} samples={SAMPLES} ... ");
        let t0 = Instant::now();
        let h = run_get_hit(m, SIZE, SAMPLES, WARMUP);
        let dur = t0.elapsed();
        let path = write_json(m, &h, overhead, SAMPLES).expect("write latency json");
        eprintln!(
            "done in {:.1}s | p50={}ns p99={}ns p999={}ns max={}ns → {}",
            dur.as_secs_f64(),
            h.value_at_quantile(0.50),
            h.value_at_quantile(0.99),
            h.value_at_quantile(0.999),
            h.max(),
            path.display()
        );
    }
}
