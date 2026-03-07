use anyhow::Result;
use ocrus_core::OcrResult;

pub fn print(result: &OcrResult) -> Result<()> {
    println!("{}", result.full_text());
    Ok(())
}
