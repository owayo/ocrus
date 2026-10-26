//! 辞書補正のメモリ確保と、モデルのロード・推論を CSV で計測する。
//! 引数は .ocnn モデルのパス。補正辞書は合成データから一時的に作る。

use ocrus_nn::ocnn::{exec::Executor, format::Model};
use ocrus_nn::tensor::NdTensor;
use ocrus_recognizer::DictCorrector;
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;
static COUNT: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
// 各確保を数え、実際のメモリ管理はシステムアロケータへ委譲する。
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: 呼び出し元から受け取ったレイアウトをそのまま渡す。
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: 同じシステムアロケータで確保したポインタとレイアウトを渡す。
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        COUNT.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(size, Ordering::Relaxed);
        // SAFETY: 元の確保情報と要求された新しいサイズをそのまま渡す。
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let usage = "usage: maintenance_profile <model.ocnn>";
    let path = std::path::Path::new(args.get(1).ok_or(usage)?);
    let mut rules = tempfile::NamedTempFile::new()?;
    writeln!(rules, "こんにちわ\tこんにちは\n仕柿\t仕事")?;
    let corrector = DictCorrector::from_file(rules.path())?;
    for repeat in [1, 64, 1024] {
        let text = "前🙂こんにちわ仕柿後".repeat(repeat);
        COUNT.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        let out = corrector.correct(black_box(&text));
        let count = COUNT.load(Ordering::Relaxed);
        let bytes = BYTES.load(Ordering::Relaxed);
        assert_eq!(out, "前🙂こんにちは仕事後".repeat(repeat));
        println!("allocation,{repeat},{count},{bytes}");
    }
    println!("model_bytes,{}", std::fs::metadata(path)?.len());
    for i in 0..11 {
        let start = Instant::now();
        let model = Model::load(path)?;
        println!("load,{i},{:.6}", start.elapsed().as_secs_f64() * 1000.0);
        black_box(model);
    }
    let exec = Executor::new(Model::load(path)?);
    for width in [104, 320] {
        for _ in 0..2 {
            black_box(exec.run(NdTensor::zeros(&[1, 3, 48, width]))?);
        }
        for i in 0..5 {
            let input = NdTensor::zeros(&[1, 3, 48, width]);
            let started = Instant::now();
            let (out, timings) = exec.run_profiled(input)?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let hash = out.data.iter().fold(0u64, |h, x| {
                h.wrapping_mul(31).wrapping_add(x.to_bits() as u64)
            });
            println!("inference,{width},{i},{ms:.6},{hash}");
            for (name, timing) in timings {
                println!("op,{width},{i},{name},{:.6}", timing.as_secs_f64() * 1000.0);
            }
        }
    }
    Ok(())
}
