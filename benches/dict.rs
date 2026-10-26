use std::hint::black_box;
use std::io::Write;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use ocrus_recognizer::DictCorrector;

fn bench_dict_correction(c: &mut Criterion) {
    let mut rules = tempfile::NamedTempFile::new().unwrap();
    writeln!(rules, "こんにちわ\tこんにちは\n仕柿\t仕事").unwrap();
    let corrector = DictCorrector::from_file(rules.path()).unwrap();
    let mut group = c.benchmark_group("dict_correct");

    // 短い認識行と長文で、一致の密度による処理時間の違いを測る。
    for repeats in [1, 64, 1024] {
        for (kind, unit) in [
            ("matches", "前🙂こんにちわ仕柿後"),
            ("no_match", "前🙂こんにちは仕事後"),
        ] {
            let text = unit.repeat(repeats);
            group.throughput(Throughput::Bytes(text.len() as u64));
            group.bench_with_input(BenchmarkId::new(kind, repeats), &text, |b, text| {
                b.iter(|| black_box(corrector.correct(black_box(text))));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_dict_correction);
criterion_main!(benches);
