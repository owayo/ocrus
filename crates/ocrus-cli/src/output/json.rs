use anyhow::Result;
use ocrus_core::OcrResult;

pub fn print(result: &OcrResult) -> Result<()> {
    let json = serde_json::to_string_pretty(result)?;
    println!("{json}");
    Ok(())
}
