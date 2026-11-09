//! Benchmarks for smoothing on long sequences.
//!
//! Run with `cargo bench`. HTML reports are written to target/criterion/report/index.html.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hmm::generic::HmmParams;
use nalgebra::{SMatrix, SVector};

const SEQUENCE_LEN: usize = 100_000;

// The 5-state, 6-symbol model from the unit tests.
fn model() -> HmmParams<5, 6> {
    HmmParams::new(
        SVector::from_row_slice(&[0.1, 0.2, 0.3, 0.15, 0.25]),
        SMatrix::from_row_slice(&[
            0.1, 0.2, 0.3, 0.15, 0.25,
            0.4, 0.1, 0.2, 0.2, 0.1,
            0.2, 0.25, 0.15, 0.3, 0.1,
            0.3, 0.1, 0.2, 0.15, 0.25,
            0.15, 0.2, 0.25, 0.3, 0.1,
        ]),
        SMatrix::from_row_slice(&[
            0.1, 0.15, 0.2, 0.25, 0.3, 0.0,
            0.3, 0.25, 0.2, 0.15, 0.1, 0.0,
            0.05, 0.1, 0.15, 0.3, 0.4, 0.0,
            0.4, 0.1, 0.15, 0.05, 0.3, 0.0,
            0.2, 0.1, 0.3, 0.15, 0.25, 0.0,
        ]),
    )
}

// A fixed pseudo-random sequence over symbols 0..5 (symbol 5 is impossible in this
// model). xorshift keeps it reproducible without a rand dependency.
fn observations(len: usize) -> Vec<usize> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 5) as usize
        })
        .collect()
}

fn smoothing(c: &mut Criterion) {
    let hmm = model();
    let y = observations(SEQUENCE_LEN);
    assert!(hmm.sequence_log_likelihood(&y).is_finite(), "benchmark sequence must be possible");

    let mut group = c.benchmark_group("smooth n=100k");
    // Report time per observation as well as per call.
    group.throughput(Throughput::Elements(SEQUENCE_LEN as u64));

    // smooth(t) runs forward over t + 1 steps and backward over n - t steps, so every
    // target costs about one full pass. These three show where the time goes:
    // mostly backward at t = 0, mostly forward at t = n - 1, an even split in the middle.
    for (label, target) in [("t=0", 0), ("t=n/2", SEQUENCE_LEN / 2), ("t=n-1", SEQUENCE_LEN - 1)] {
        group.bench_with_input(BenchmarkId::from_parameter(label), &target, |b, &t| {
            b.iter(|| hmm.smooth(black_box(&y), black_box(t)))
        });
    }

    // Every t at once: one forward pass plus one backward pass, instead of one full
    // pass per target.
    group.bench_function("smooth_all", |b| b.iter(|| hmm.smooth_all(black_box(&y))));

    // The two passes on their own, for comparison.
    group.bench_function("forward pass", |b| {
        b.iter(|| hmm.forward(black_box(&y)).last())
    });
    group.bench_function("backward pass", |b| {
        b.iter(|| hmm.backward(black_box(&y)).last())
    });

    group.finish();
}

criterion_group!(benches, smoothing);
criterion_main!(benches);
