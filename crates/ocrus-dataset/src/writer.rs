use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use image::GrayImage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DatasetStats {
    pub total_images: u64,
    pub train_images: u64,
    pub val_images: u64,
    pub categories: Vec<String>,
    pub fonts_used: Vec<String>,
    pub augmentations_applied: Vec<String>,
    pub per_category: HashMap<String, u64>,
    pub per_font: HashMap<String, u64>,
    #[serde(skip)]
    pub elapsed: Duration,
    // Aliases for backward compat
    pub total_samples: u64,
    pub train_samples: u64,
    pub val_samples: u64,
}

pub struct DatasetWriter {
    output_dir: PathBuf,
    samples_dir: PathBuf,
    counter: AtomicU64,
    labels: std::sync::Mutex<Vec<LabelEntry>>,
}

#[derive(Debug)]
struct LabelEntry {
    filename: String,
    label: String,
    category: String,
    font: String,
    augment_type: String,
}

impl DatasetWriter {
    pub fn new(output_dir: &Path) -> Result<Self> {
        let samples_dir = output_dir.join("samples");
        fs::create_dir_all(&samples_dir)
            .with_context(|| format!("failed to create samples dir: {}", samples_dir.display()))?;
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
            samples_dir,
            counter: AtomicU64::new(0),
            labels: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn add_sample(
        &self,
        image: &GrayImage,
        label: &str,
        category: &str,
        font: &str,
        augment: &str,
    ) -> Result<()> {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed);
        let filename = format!("{idx:06}.png");
        let path = self.samples_dir.join(&filename);
        image
            .save(&path)
            .with_context(|| format!("failed to save sample: {}", path.display()))?;
        self.labels.lock().unwrap().push(LabelEntry {
            filename,
            label: label.to_string(),
            category: category.to_string(),
            font: font.to_string(),
            augment_type: augment.to_string(),
        });
        Ok(())
    }

    pub fn finish(self, val_ratio: f32) -> Result<DatasetStats> {
        let labels = self.labels.into_inner().unwrap();
        let total = labels.len() as u64;
        let val_count = (total as f32 * val_ratio) as u64;
        let train_count = total - val_count;

        // Write labels.tsv
        let tsv_path = self.output_dir.join("labels.tsv");
        let mut tsv = fs::File::create(&tsv_path)?;
        writeln!(tsv, "filename\tlabel\tcategory\tfont\taugment_type")?;
        let mut categories = std::collections::HashSet::new();
        let mut fonts = std::collections::HashSet::new();
        let mut augmentations = std::collections::HashSet::new();
        let mut per_category: HashMap<String, u64> = HashMap::new();
        let mut per_font: HashMap<String, u64> = HashMap::new();
        for entry in &labels {
            writeln!(
                tsv,
                "{}\t{}\t{}\t{}\t{}",
                entry.filename, entry.label, entry.category, entry.font, entry.augment_type
            )?;
            categories.insert(entry.category.clone());
            fonts.insert(entry.font.clone());
            augmentations.insert(entry.augment_type.clone());
            *per_category.entry(entry.category.clone()).or_default() += 1;
            *per_font.entry(entry.font.clone()).or_default() += 1;
        }

        let stats = DatasetStats {
            total_images: total,
            train_images: train_count,
            val_images: val_count,
            total_samples: total,
            train_samples: train_count,
            val_samples: val_count,
            categories: categories.into_iter().collect(),
            fonts_used: fonts.into_iter().collect(),
            augmentations_applied: augmentations.into_iter().collect(),
            per_category,
            per_font,
            elapsed: Duration::default(),
        };

        // Write manifest.json
        let manifest_path = self.output_dir.join("manifest.json");
        let manifest = serde_json::json!({
            "version": "1.0",
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            "stats": &stats,
        });
        let manifest_file = fs::File::create(&manifest_path)?;
        serde_json::to_writer_pretty(manifest_file, &manifest)?;

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn writer_creates_files() {
        let tmp =
            std::env::temp_dir().join(format!("ocrus_dataset_test_writer_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let writer = DatasetWriter::new(&tmp).unwrap();
        let img = GrayImage::from_pixel(64, 48, Luma([128u8]));
        writer
            .add_sample(&img, "あ", "hiragana", "TestFont", "original")
            .unwrap();
        let stats = writer.finish(0.0).unwrap();
        assert_eq!(stats.total_images, 1);
        assert!(tmp.join("samples/000000.png").exists());
        assert!(tmp.join("labels.tsv").exists());
        assert!(tmp.join("manifest.json").exists());
        let _ = fs::remove_dir_all(&tmp);
    }

    #[test]
    fn writer_multi_char_label() {
        let tmp =
            std::env::temp_dir().join(format!("ocrus_dataset_test_multi_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let writer = DatasetWriter::new(&tmp).unwrap();
        let img = GrayImage::from_pixel(192, 48, Luma([128u8]));
        writer
            .add_sample(&img, "あいう", "hiragana", "TestFont", "original")
            .unwrap();
        let stats = writer.finish(0.0).unwrap();
        assert_eq!(stats.total_images, 1);
        let tsv = fs::read_to_string(tmp.join("labels.tsv")).unwrap();
        let data_line = tsv.lines().nth(1).unwrap();
        assert!(
            data_line.contains("あいう"),
            "label should be multi-char string"
        );
        let _ = fs::remove_dir_all(&tmp);
    }
}
