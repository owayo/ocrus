//! Python bindings for the ocrus OCR engine.
//!
//! This crate is the thin translation layer only: it converts Python values to Rust,
//! calls [`ocrus_engine`], and converts results back. Anything that could be described as
//! OCR logic belongs in `ocrus-engine` so that the CLI and Python see identical behaviour.
//!
//! The compiled module is imported as `ocrus._native`; the user-facing API lives in the
//! `python/ocrus/__init__.py` wrapper, which adds numpy support and docstrings without
//! making numpy a build dependency of this crate.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use ocrus_core::{
    BBox, CharsetMode, EngineConfigBuilder, OcrMode, OcrResult, OcrusError, Page, RubyAnnotation,
    TextLine,
};
use ocrus_engine::{DICT_FILE, OcrEngine, REC_MODEL_FILE};

create_exception!(
    ocrus,
    OcrusPyError,
    PyException,
    "Base class for ocrus errors."
);
create_exception!(
    ocrus,
    ModelNotFoundError,
    OcrusPyError,
    "The model directory does not contain rec.ocnn and dict.txt."
);
create_exception!(
    ocrus,
    ModelError,
    OcrusPyError,
    "The model exists but could not be loaded or executed."
);
create_exception!(
    ocrus,
    ImageError,
    OcrusPyError,
    "The image could not be read or decoded."
);
create_exception!(
    ocrus,
    ConfigError,
    OcrusPyError,
    "The engine configuration is not usable."
);

/// Map an engine error onto the exception a Python caller can act on.
///
/// The distinction that matters in practice is "you have not downloaded the model yet"
/// versus everything else, so a missing file gets its own type with a recovery hint.
fn to_py_err(err: OcrusError) -> PyErr {
    match err {
        OcrusError::Model(msg) if msg.starts_with("model file not found") => {
            ModelNotFoundError::new_err(msg)
        }
        OcrusError::Model(msg) => ModelError::new_err(msg),
        OcrusError::Image(msg) => ImageError::new_err(msg),
        OcrusError::Config(msg) => ConfigError::new_err(msg),
        other => OcrusPyError::new_err(other.to_string()),
    }
}

#[pyclass(name = "BBox", module = "ocrus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyBBox {
    #[pyo3(get)]
    pub x: u32,
    #[pyo3(get)]
    pub y: u32,
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
}

#[pymethods]
impl PyBBox {
    /// `(x, y, width, height)`, handy for slicing images with PIL or numpy.
    fn as_tuple(&self) -> (u32, u32, u32, u32) {
        (self.x, self.y, self.width, self.height)
    }

    fn __repr__(&self) -> String {
        format!(
            "BBox(x={}, y={}, width={}, height={})",
            self.x, self.y, self.width, self.height
        )
    }
}

impl From<&BBox> for PyBBox {
    fn from(b: &BBox) -> Self {
        Self {
            x: b.x,
            y: b.y,
            width: b.width,
            height: b.height,
        }
    }
}

#[pyclass(name = "RubyAnnotation", module = "ocrus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRubyAnnotation {
    #[pyo3(get)]
    pub ruby_text: String,
    #[pyo3(get)]
    pub bbox: PyBBox,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl PyRubyAnnotation {
    fn __repr__(&self) -> String {
        format!(
            "RubyAnnotation(ruby_text={:?}, bbox={}, confidence={:.3})",
            self.ruby_text,
            self.bbox.__repr__(),
            self.confidence
        )
    }
}

impl From<&RubyAnnotation> for PyRubyAnnotation {
    fn from(r: &RubyAnnotation) -> Self {
        Self {
            ruby_text: r.ruby_text.clone(),
            bbox: PyBBox::from(&r.bbox),
            confidence: r.confidence,
        }
    }
}

#[pyclass(name = "TextLine", module = "ocrus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyTextLine {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub bbox: PyBBox,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub ruby: Vec<PyRubyAnnotation>,
}

#[pymethods]
impl PyTextLine {
    fn __repr__(&self) -> String {
        format!(
            "TextLine(text={:?}, bbox={}, confidence={:.3})",
            self.text,
            self.bbox.__repr__(),
            self.confidence
        )
    }

    fn __str__(&self) -> String {
        self.text.clone()
    }
}

impl From<&TextLine> for PyTextLine {
    fn from(line: &TextLine) -> Self {
        Self {
            text: line.text.clone(),
            bbox: PyBBox::from(&line.bbox),
            confidence: line.confidence,
            ruby: line.ruby.iter().map(PyRubyAnnotation::from).collect(),
        }
    }
}

#[pyclass(name = "Page", module = "ocrus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPage {
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    #[pyo3(get)]
    pub lines: Vec<PyTextLine>,
}

#[pymethods]
impl PyPage {
    /// The page text, one line per detected line.
    fn text(&self) -> String {
        self.lines
            .iter()
            .map(|l| l.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn __repr__(&self) -> String {
        format!(
            "Page(width={}, height={}, lines={})",
            self.width,
            self.height,
            self.lines.len()
        )
    }
}

impl From<&Page> for PyPage {
    fn from(page: &Page) -> Self {
        Self {
            width: page.width,
            height: page.height,
            lines: page.lines.iter().map(PyTextLine::from).collect(),
        }
    }
}

#[pyclass(name = "OcrResult", module = "ocrus", frozen)]
pub struct PyOcrResult {
    inner: OcrResult,
}

#[pymethods]
impl PyOcrResult {
    #[getter]
    fn pages(&self) -> Vec<PyPage> {
        self.inner.pages.iter().map(PyPage::from).collect()
    }

    /// Every line of every page, joined with newlines.
    fn full_text(&self) -> String {
        self.inner.full_text()
    }

    /// The same JSON the CLI prints with `--format json`.
    ///
    /// Serialising from the Rust result rather than from the Python objects is what keeps
    /// the two outputs from drifting apart.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner)
            .map_err(|e| OcrusPyError::new_err(format!("failed to serialize result: {e}")))
    }

    /// The JSON structure as Python dicts and lists.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let json = self.to_json()?;
        PyModule::import(py, "json")?.call_method1("loads", (json,))
    }

    fn __len__(&self) -> usize {
        self.inner.pages.len()
    }

    fn __str__(&self) -> String {
        self.inner.full_text()
    }

    fn __repr__(&self) -> String {
        let lines: usize = self.inner.pages.iter().map(|p| p.lines.len()).sum();
        format!(
            "OcrResult(pages={}, lines={})",
            self.inner.pages.len(),
            lines
        )
    }
}

impl From<OcrResult> for PyOcrResult {
    fn from(inner: OcrResult) -> Self {
        Self { inner }
    }
}

/// A loaded OCR engine. Construct once, reuse for every image.
#[pyclass(name = "OcrEngine", module = "ocrus")]
pub struct PyOcrEngine {
    inner: Arc<OcrEngine>,
    model_dir: PathBuf,
}

#[pymethods]
impl PyOcrEngine {
    #[new]
    #[pyo3(signature = (
        *,
        model_dir = None,
        mode = "auto",
        charset = "full",
        dict_path = None,
        threads = None,
        ruby = false,
        beam_width = 5,
        confidence_threshold = 0.5,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model_dir: Option<PathBuf>,
        mode: &str,
        charset: &str,
        dict_path: Option<PathBuf>,
        threads: Option<usize>,
        ruby: bool,
        beam_width: usize,
        confidence_threshold: f32,
    ) -> PyResult<Self> {
        let mut builder = EngineConfigBuilder::new();

        if let Some(dir) = model_dir {
            builder = builder.model_dir(dir);
        }
        builder = builder.mode(match mode {
            "auto" => OcrMode::Auto,
            "fastest" => OcrMode::Fastest,
            "accurate" => OcrMode::Accurate,
            other => {
                return Err(ConfigError::new_err(format!(
                    "unknown mode {other:?}: expected 'auto', 'fastest' or 'accurate'"
                )));
            }
        });
        builder = builder.charset(match charset {
            "full" => CharsetMode::Full,
            "jis" => CharsetMode::Jis,
            other => {
                return Err(ConfigError::new_err(format!(
                    "unknown charset {other:?}: expected 'full' or 'jis'"
                )));
            }
        });
        if let Some(path) = dict_path {
            builder = builder.dict_path(path);
        }
        if let Some(n) = threads {
            builder = builder.num_threads(n);
        }
        builder = builder
            .ruby_separation(ruby)
            .beam_width(beam_width)
            .confidence_threshold(confidence_threshold);

        let config = builder.build();
        let model_dir = config.model_dir.clone();
        let engine = OcrEngine::new(config).map_err(to_py_err)?;

        Ok(Self {
            inner: Arc::new(engine),
            model_dir,
        })
    }

    /// The model directory this engine loaded from.
    #[getter]
    fn model_dir(&self) -> PathBuf {
        self.model_dir.clone()
    }

    /// Recognize an image file.
    fn recognize_path(&self, py: Python<'_>, path: PathBuf) -> PyResult<PyOcrResult> {
        let engine = Arc::clone(&self.inner);
        // Inference takes seconds and touches no Python state, so other threads should run.
        let result = py
            .detach(move || engine.recognize_path(&path))
            .map_err(to_py_err)?;
        Ok(result.into())
    }

    /// Recognize an encoded image (PNG, JPEG, ...) held in memory.
    fn recognize_bytes(&self, py: Python<'_>, data: Vec<u8>) -> PyResult<PyOcrResult> {
        let engine = Arc::clone(&self.inner);
        let result = py
            .detach(move || engine.recognize_bytes(&data))
            .map_err(to_py_err)?;
        Ok(result.into())
    }

    /// Recognize raw pixels: `width * height * channels` bytes, row-major.
    #[pyo3(signature = (data, width, height, channels = 1))]
    fn recognize_raw(
        &self,
        py: Python<'_>,
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> PyResult<PyOcrResult> {
        let engine = Arc::clone(&self.inner);
        let result = py
            .detach(move || engine.recognize_raw(&data, width, height, channels))
            .map_err(to_py_err)?;
        Ok(result.into())
    }

    fn __repr__(&self) -> String {
        format!("OcrEngine(model_dir={:?})", self.model_dir)
    }
}

/// The model directory used when none is given: `$OCRUS_MODEL_DIR`, else `~/.ocrus/models`.
#[pyfunction]
fn default_model_dir() -> PathBuf {
    ocrus_engine::default_model_dir()
}

/// Whether a model directory holds both files the engine needs.
#[pyfunction]
#[pyo3(signature = (model_dir = None))]
fn models_ready(model_dir: Option<PathBuf>) -> bool {
    let dir = model_dir.unwrap_or_else(ocrus_engine::default_model_dir);
    ocrus_engine::models_ready(&dir)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("REC_MODEL_FILE", REC_MODEL_FILE)?;
    m.add("DICT_FILE", DICT_FILE)?;

    m.add_class::<PyBBox>()?;
    m.add_class::<PyRubyAnnotation>()?;
    m.add_class::<PyTextLine>()?;
    m.add_class::<PyPage>()?;
    m.add_class::<PyOcrResult>()?;
    m.add_class::<PyOcrEngine>()?;

    m.add("OcrusError", py.get_type::<OcrusPyError>())?;
    m.add("ModelNotFoundError", py.get_type::<ModelNotFoundError>())?;
    m.add("ModelError", py.get_type::<ModelError>())?;
    m.add("ImageError", py.get_type::<ImageError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;

    m.add_function(wrap_pyfunction!(default_model_dir, m)?)?;
    m.add_function(wrap_pyfunction!(models_ready, m)?)?;

    Ok(())
}
