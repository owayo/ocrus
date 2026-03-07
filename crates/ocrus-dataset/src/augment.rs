use image::{GrayImage, Luma};
use imageproc::filter::gaussian_blur_f32;
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use imageproc::noise::salt_and_pepper_noise;
use rand::{Rng, RngExt};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AugmentType {
    Original,
    Rotate(f32),
    Blur(f32),
    Noise(f64),
    Contrast(f32),
}

impl AugmentType {
    pub fn label(&self) -> String {
        match self {
            Self::Original => "original".to_string(),
            Self::Rotate(deg) => format!("rotate_{deg:.1}"),
            Self::Blur(sigma) => format!("blur_{sigma:.1}"),
            Self::Noise(rate) => format!("noise_{rate:.2}"),
            Self::Contrast(factor) => format!("contrast_{factor:.1}"),
        }
    }
}

pub fn apply_augmentation(img: &GrayImage, aug: &AugmentType, rng: &mut impl Rng) -> GrayImage {
    match aug {
        AugmentType::Original => img.clone(),
        AugmentType::Rotate(deg) => augment_rotate(img, *deg),
        AugmentType::Blur(sigma) => augment_blur(img, *sigma),
        AugmentType::Noise(rate) => augment_noise(img, *rate, rng),
        AugmentType::Contrast(factor) => augment_contrast(img, *factor),
    }
}

pub fn augment_rotate(img: &GrayImage, angle_deg: f32) -> GrayImage {
    let theta = angle_deg.to_radians();
    rotate_about_center(img, theta, Interpolation::Bilinear, Luma([255u8]))
}

pub fn augment_blur(img: &GrayImage, sigma: f32) -> GrayImage {
    gaussian_blur_f32(img, sigma)
}

pub fn augment_noise(img: &GrayImage, rate: f64, rng: &mut impl Rng) -> GrayImage {
    let seed = rng.random::<u64>();
    salt_and_pepper_noise(img, rate, seed)
}

pub fn augment_contrast(img: &GrayImage, factor: f32) -> GrayImage {
    let mut out = img.clone();
    for pixel in out.pixels_mut() {
        let v = pixel[0] as f32;
        let adjusted = ((v - 128.0) * factor + 128.0).clamp(0.0, 255.0);
        pixel[0] = adjusted as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_image() -> GrayImage {
        GrayImage::from_pixel(64, 48, Luma([128u8]))
    }

    #[test]
    fn augment_rotate_no_panic() {
        let img = test_image();
        let out = augment_rotate(&img, 2.0);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn augment_blur_no_panic() {
        let img = test_image();
        let out = augment_blur(&img, 1.0);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn augment_noise_no_panic() {
        let img = test_image();
        let mut rng = rand::rng();
        let out = augment_noise(&img, 0.05, &mut rng);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn augment_contrast_no_panic() {
        let img = test_image();
        let out = augment_contrast(&img, 1.5);
        assert_eq!(out.dimensions(), img.dimensions());
    }

    #[test]
    fn apply_original_is_identity() {
        let img = test_image();
        let mut rng = rand::rng();
        let out = apply_augmentation(&img, &AugmentType::Original, &mut rng);
        assert_eq!(out, img);
    }
}
