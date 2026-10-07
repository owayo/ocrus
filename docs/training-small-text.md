# Screenshot and small-text training and evaluation

## Data generation support

- `dataset generate` and font-rendering `dataset from-failures` accept additional font directories.
- Recursive discovery includes TTF, OTF, TTC and OTC files, excluding duplicate search paths and symbolic-link loops.
- Each TTC / OTC face is used individually, sharing the file data between faces.
- Style classification prefers each face's family name: `sans-serif` is gothic and `sans mono` is monospace.
- Characters missing from a font are excluded before rendering, preventing a missing-glyph box from receiving another character's label.
- `--render-heights` renders smaller images and saves them at their native resolution.

Recognition accuracy has not been measured for these changes. Their effect requires retraining, export, `.ocnn` conversion and comparison through the production pipeline.

## Font directories and rendering sizes

```bash
cargo run --release -p ocrus-cli -- dataset generate \
  --output ./training_data/small-text \
  --font-dirs ./fonts/screen \
  --no-system-fonts \
  --font-styles gothic,monospace \
  --categories hiragana,katakana,joyo_kanji,halfwidth_alnum,halfwidth_symbols,fullwidth_alnum,fullwidth_symbols \
  --chars-per-image 15 \
  --samples-per-char 1 \
  --render-heights 12,16,20,24,32,48
```

`--font-dirs` adds directories to the system font search and may be repeated. `--no-system-fonts` restricts discovery to the supplied directories, which is useful for fixing a training set. Missing additional directories produce an error.

`--render-heights` sets the height of the entire image canvas, rather than CSS `font-size` or glyph ink height. Glyphs occupy less than the full canvas. Display scaling and Retina rendering affect screenshot pixel sizes, so match the heights to actual text lines. The default is 48px. More heights produce more images; begin with a small selection of fonts and categories.

Noise, blur and other augmentations are applied at the native rendering resolution. Resizing to model input happens during training. The [PaddleOCR configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml) uses multiple training input sizes and a height of 48px for evaluation. Resizing during generation would add another interpolation stage and change the distribution.

Rendering at small sizes captures thin strokes merging and pixel rounding that blurring a large glyph cannot reproduce. `ab_glyph` does not fully reproduce OS or browser hinting and subpixel rendering. Include actual screenshots in evaluation.

The first face in TTC / OTC files retains the file name; later faces use `filename#1`, `filename#2` and so on. Collisions between files receive suffixes such as `#file2`. Discovery order is fixed by file path. Identical font contents copied into separate files are not deduplicated; organize those copies yourself. Variable fonts do not generate multiple weights automatically. Supply static Regular, Medium or Bold fonts for different weights.

## General font selection criteria

Choose gothic, UD or monospace styles and Regular-to-Bold weights according to the target use.
Check language coverage, rendering conditions, shared glyph designs and the permissions for the fonts and generated data.
Keep experiment-specific font names, acquisition sources, license reviews and split assignments with the data-management records.
This public guide describes general selection criteria and the dataset APIs.

Follow each font's license for use and redistribution. Check font-level counts in `labels.tsv` and `manifest.json` so one family does not dominate merely because more fonts were added.

## Evaluation plan

### Reproducible UI line evaluation

`scripts/src/ocrus_scripts/evaluate_ui.py` renders light and dark UI labels at
12, 16, 24 and 32px, both with compact margins and on a 1,200px-wide canvas.
Supply local Japanese font files. Use different font families for the `dev` and
`test` splits; those splits also use different strings. This is a synthetic
regression corpus, rather than a measurement of actual OS screenshots.

```bash
uv run --project scripts --with pillow python scripts/src/ocrus_scripts/evaluate_ui.py render \
  --fonts /path/to/font1.ttf /path/to/font2.ttf --split dev --output logs/ui-dev
cargo build --locked --release -p ocrus-engine --example recognize_batch
uv run --project scripts python scripts/src/ocrus_scripts/evaluate_ui.py evaluate \
  --manifest logs/ui-dev/manifest.json --binary target/release/examples/recognize_batch \
  --models /path/to/models --output logs/ui-before.json
```

After changing the engine, rebuild the example and evaluate the **same manifest**
into another report. On Windows the example binary has an `.exe` suffix. Reports
include input, model, dictionary and executable hashes, strict CER, substitutions,
deletions, insertions and exact line matches, grouped by font, size, theme and
margin. No character normalization is applied. Keep the weights and dictionary
fixed when evaluating a pipeline change, and inspect every group for regressions.

Single-character tests cannot establish practical accuracy for small UI text. Evaluate text lines under these conditions:

1. **Use the production `OcrEngine`.** Follow the same recognition path as CLI and Python, recording `CER = (substitutions + deletions + insertions) / ground-truth characters` and exact line match rate.
2. **Exclude evaluation fonts from training.** Split related weights and proportional or monospace derivatives by family. Use actual screenshots as a separate evaluation set.
3. **Report results by size.** Aggregate numbers can conceal regressions at normal sizes. Treat half-width/full-width and symbol distinctions as primary metrics; normalized results are secondary.
4. **Include real text.** Current category strings list characters in sequence. Add natural UI strings such as 「ファイルを保存する」, 「検索結果：128件」 and `v2.4.1`, including mixed Japanese/English, paths, URLs, numbers and symbols.
5. **Include display conditions.** Evaluate light and dark backgrounds, text colors, thin strokes, scaling, resizing, JPEG compression and actual OS rendering.

The Rust `val_ratio` currently calculates counts without splitting images into separate validation files. Python `finetune` independently makes a random 90/10 split of `labels.tsv`, so augmented copies of the same font and string can enter both sets. This score does not measure generalization to unknown fonts. Split by font family and original image, then have Python consume that split unchanged.

`todo.md` still records an unresolved post-training export issue. Before replacing the default model, fix it and verify exported outputs and `.ocnn` golden results. AI agents may also run training, large dataset generation and model conversion. Run long tasks independently of the agent session, preserve logs and progress, and verify completion or failure.
