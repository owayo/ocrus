//! Small, versioned C boundary. Recognition stays in `ocrus-engine`.
//! See `include/ocrus.h` for pointer lifetimes and ownership.

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

use ocrus_core::{EngineConfig, OcrMode};
use ocrus_engine::OcrEngine;

pub struct OcrusEngine {
    engine: OcrEngine,
    pool: rayon::ThreadPool,
    poisoned: AtomicBool,
}

pub struct OcrusCancel(AtomicBool);

const INVALID: i32 = 1;
const ENGINE_ERROR: i32 = 2;
const CANCELLED: i32 = 3;
const PANIC: i32 = 4;
type FfiResult = Result<(), (i32, String)>;

#[unsafe(no_mangle)]
pub extern "C" fn ocrus_abi_version() -> u32 {
    1
}

fn owned_string(value: String) -> *mut c_char {
    // Interior NULs can occur in OS error messages or model metadata.
    CString::new(value.replace('\0', "\\0"))
        .expect("interior NULs were escaped")
        .into_raw()
}

/// # Safety
/// `error` is a writable, initially null output slot (or null).
unsafe fn boundary(error: *mut *mut c_char, body: impl FnOnce() -> FfiResult) -> i32 {
    if !error.is_null() {
        unsafe { *error = ptr::null_mut() };
    }
    let result = catch_unwind(AssertUnwindSafe(body));
    let (status, message) = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(failure)) => failure,
        Err(_) => (PANIC, "OCR engine panicked; recreate the engine".into()),
    };
    if !error.is_null() {
        unsafe { *error = owned_string(message) };
    }
    status
}

/// # Safety
/// `model_dir` is a valid NUL-terminated UTF-8 string. Output slots are writable
/// and initially null. The returned handle must be freed exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_engine_create(
    model_dir: *const c_char,
    mode: u32,
    threads: u32,
    out_engine: *mut *mut OcrusEngine,
    out_error: *mut *mut c_char,
) -> i32 {
    if !out_engine.is_null() {
        unsafe { *out_engine = ptr::null_mut() };
    }
    unsafe {
        boundary(out_error, || {
            if model_dir.is_null() || out_engine.is_null() || !(1..=64).contains(&threads) {
                return Err((
                    INVALID,
                    "invalid model directory, output slot or thread count".into(),
                ));
            }
            let dir = CStr::from_ptr(model_dir)
                .to_str()
                .map_err(|_| (INVALID, "model directory is not UTF-8".into()))?;
            let mode = match mode {
                0 => OcrMode::Auto,
                1 => OcrMode::Accurate,
                _ => return Err((INVALID, "mode must be 0 or 1".into())),
            };
            let config = EngineConfig {
                model_dir: PathBuf::from(dir),
                mode,
                num_threads: threads as usize,
                ..EngineConfig::default()
            };
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads as usize)
                .build()
                .map_err(|e| (ENGINE_ERROR, e.to_string()))?;
            let engine = OcrEngine::new(config).map_err(|e| (ENGINE_ERROR, e.to_string()))?;
            *out_engine = Box::into_raw(Box::new(OcrusEngine {
                engine,
                pool,
                poisoned: AtomicBool::new(false),
            }));
            Ok(())
        })
    }
}

/// # Safety
/// `engine` is null or a live handle returned by `ocrus_engine_create`, with no
/// active calls. It must not be used again after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_engine_free(engine: *mut OcrusEngine) {
    if !engine.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe { drop(Box::from_raw(engine)) }));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ocrus_cancel_create() -> *mut OcrusCancel {
    catch_unwind(|| Box::into_raw(Box::new(OcrusCancel(AtomicBool::new(false)))))
        .unwrap_or(ptr::null_mut())
}

/// # Safety
/// `cancel` is null or live, and remains live throughout this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_cancel_request(cancel: *const OcrusCancel) {
    if let Some(cancel) = unsafe { cancel.as_ref() } {
        cancel.0.store(true, Ordering::Relaxed);
    }
}

/// # Safety
/// `cancel` is null or a live token with no active calls or recognition using it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_cancel_free(cancel: *mut OcrusCancel) {
    if !cancel.is_null() {
        unsafe { drop(Box::from_raw(cancel)) };
    }
}

fn rgba_pixels(
    pixels: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    format: u32,
) -> Result<image::DynamicImage, (i32, String)> {
    let row = (width as usize).checked_mul(4);
    let size = stride.checked_mul(height as usize);
    let count = (width as u64).checked_mul(height as u64);
    if width == 0
        || height == 0
        || count.is_none_or(|n| n > 40_000_000)
        || row.is_none_or(|r| stride < r)
        || size.is_none_or(|s| pixels.len() < s)
        || format > 1
    {
        return Err((
            INVALID,
            "invalid pixel dimensions, stride, format or length".into(),
        ));
    }
    let row = row.unwrap();
    let mut packed = Vec::with_capacity(row * height as usize);
    for y in 0..height as usize {
        packed.extend_from_slice(&pixels[y * stride..y * stride + row]);
    }
    if format == 1 {
        for pixel in packed.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
        }
    }
    image::RgbaImage::from_raw(width, height, packed)
        .map(image::DynamicImage::ImageRgba8)
        .ok_or_else(|| (INVALID, "could not build pixel image".into()))
}

/// # Safety
/// All non-null pointers are valid for the duration of this call. `pixels`
/// covers `len` readable bytes (at most isize::MAX). Output slots are writable
/// and initially null. The engine is not concurrently used or freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_recognize_pixels(
    engine: *const OcrusEngine,
    pixels: *const u8,
    len: usize,
    width: u32,
    height: u32,
    stride: usize,
    format: u32,
    cancel: *const OcrusCancel,
    out_json: *mut *mut c_char,
    out_error: *mut *mut c_char,
) -> i32 {
    if !out_json.is_null() {
        unsafe { *out_json = ptr::null_mut() };
    }
    let status = unsafe {
        boundary(out_error, || {
            if engine.is_null()
                || pixels.is_null()
                || out_json.is_null()
                || len > isize::MAX as usize
            {
                return Err((INVALID, "null handle, pixels or output slot".into()));
            }
            let engine = &*engine;
            if engine.poisoned.load(Ordering::Relaxed) {
                return Err((PANIC, "engine is poisoned; recreate it".into()));
            }
            let cancel = cancel.as_ref();
            let cancelled = || cancel.is_some_and(|c| c.0.load(Ordering::Relaxed));
            if cancelled() {
                return Err((CANCELLED, "recognition cancelled".into()));
            }
            let image = rgba_pixels(
                std::slice::from_raw_parts(pixels, len),
                width,
                height,
                stride,
                format,
            )?;
            let result = engine
                .pool
                .install(|| engine.engine.recognize_image_cancellable(&image, cancelled));
            if cancelled() {
                return Err((CANCELLED, "recognition cancelled".into()));
            }
            let result = result.map_err(|e| (ENGINE_ERROR, e.to_string()))?;
            let json = serde_json::to_string(&result).map_err(|e| (ENGINE_ERROR, e.to_string()))?;
            *out_json = owned_string(json);
            Ok(())
        })
    };
    if status == PANIC && !engine.is_null() {
        unsafe { (*engine).poisoned.store(true, Ordering::Relaxed) };
    }
    status
}

/// # Safety
/// `string` is null or a live string returned by this library. Free exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ocrus_string_free(string: *mut c_char) {
    if !string.is_null() {
        unsafe { drop(CString::from_raw(string)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_stride_is_removed_without_touching_padding() {
        let image = rgba_pixels(&[3, 2, 1, 255, 99, 99, 6, 5, 4, 255, 99, 99], 1, 2, 6, 1).unwrap();
        assert_eq!(image.to_rgba8().into_raw(), [1, 2, 3, 255, 4, 5, 6, 255]);
    }

    #[test]
    fn dimensions_and_format_are_checked_before_allocation() {
        for (w, h, stride, format) in [
            (0, 1, 4, 0),
            (1, 2, 4, 0),
            (1, 1, 3, 0),
            (1, 1, 4, 2),
            (u32::MAX, u32::MAX, usize::MAX, 0),
        ] {
            assert!(rgba_pixels(&[0; 4], w, h, stride, format).is_err());
        }
    }

    #[test]
    fn null_inputs_clear_output_slots_and_return_owned_errors() {
        let mut engine = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe { ocrus_engine_create(ptr::null(), 0, 1, &mut engine, &mut error) };
        assert_eq!(status, INVALID);
        assert!(engine.is_null());
        assert!(!error.is_null());
        unsafe { ocrus_string_free(error) };
    }

    #[test]
    fn panic_is_contained_and_cancellation_is_thread_safe() {
        let mut error = ptr::null_mut();
        assert_eq!(unsafe { boundary(&mut error, || panic!("test")) }, PANIC);
        unsafe { ocrus_string_free(error) };
        let token = ocrus_cancel_create();
        assert!(!token.is_null());
        unsafe {
            ocrus_cancel_request(token);
            assert!((*token).0.load(Ordering::Relaxed));
            ocrus_cancel_free(token);
            ocrus_cancel_free(ptr::null_mut());
            ocrus_engine_free(ptr::null_mut());
            ocrus_string_free(ptr::null_mut());
        }
    }
}
