use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use anyhow::Result;
use ratatui::Frame;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, List, ListItem, ListState, Paragraph, Wrap};

struct MenuItem {
    label: &'static str,
    description: &'static str,
    command_fn: fn(&Paths) -> Vec<String>,
}

struct Paths {
    workspace_root: PathBuf,
    model_dir: PathBuf,
}

const MENU_ITEMS: &[MenuItem] = &[
    MenuItem {
        label: "E2E Accuracy Test",
        description: "Run char_accuracy test across all fonts (slow, ~10min)",
        command_fn: |_| {
            vec![
                "cargo".into(),
                "test".into(),
                "-p".into(),
                "ocrus-cli".into(),
                "--test".into(),
                "char_accuracy".into(),
                "--".into(),
                "--ignored".into(),
                "--nocapture".into(),
            ]
        },
    },
    MenuItem {
        label: "Download Models",
        description: "Download PP-OCRv5 ONNX model and dict, then convert to .ocnn",
        command_fn: |p| {
            vec![
                "bash".into(),
                p.workspace_root
                    .join("models/download.sh")
                    .to_string_lossy()
                    .into(),
            ]
        },
    },
    MenuItem {
        label: "ONNX -> .ocnn Convert",
        description: "Convert rec.onnx to rec.ocnn (pure Rust inference format)",
        command_fn: |p| {
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                p.workspace_root.join("scripts").to_string_lossy().into(),
                "python".into(),
                p.workspace_root
                    .join("scripts/src/ocrus_scripts/convert_to_ocnn.py")
                    .to_string_lossy()
                    .into(),
                p.model_dir.join("rec.onnx").to_string_lossy().into(),
                "-o".into(),
                p.model_dir.join("rec.ocnn").to_string_lossy().into(),
            ]
        },
    },
    MenuItem {
        label: "Dataset Generate",
        description: "Generate training data from system fonts",
        command_fn: |p| {
            vec![
                "cargo".into(),
                "run".into(),
                "-p".into(),
                "ocrus-cli".into(),
                "--".into(),
                "dataset".into(),
                "generate".into(),
                "--output".into(),
                p.workspace_root
                    .join("training_data")
                    .to_string_lossy()
                    .into(),
            ]
        },
    },
    MenuItem {
        label: "Fine-tune",
        description: "Fine-tune PP-OCRv5 recognition model (requires PaddlePaddle)",
        command_fn: |p| {
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                p.workspace_root.join("scripts").to_string_lossy().into(),
                "finetune".into(),
            ]
        },
    },
    MenuItem {
        label: "Export ONNX",
        description: "Export fine-tuned model to ONNX format",
        command_fn: |p| {
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                p.workspace_root.join("scripts").to_string_lossy().into(),
                "export-onnx".into(),
            ]
        },
    },
    MenuItem {
        label: "Quantize (INT8)",
        description: "Quantize ONNX model to INT8 for faster inference",
        command_fn: |p| {
            vec![
                "uv".into(),
                "run".into(),
                "--project".into(),
                p.workspace_root.join("scripts").to_string_lossy().into(),
                "quantize".into(),
            ]
        },
    },
    MenuItem {
        label: "Benchmark",
        description: "Run inference benchmarks",
        command_fn: |_| vec!["cargo".into(), "bench".into()],
    },
];

struct App {
    list_state: ListState,
    status: Option<String>,
}

impl App {
    fn new() -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        Self {
            list_state,
            status: None,
        }
    }

    fn selected(&self) -> usize {
        self.list_state.selected().unwrap_or(0)
    }

    fn next(&mut self) {
        let i = (self.selected() + 1) % MENU_ITEMS.len();
        self.list_state.select(Some(i));
    }

    fn previous(&mut self) {
        let i = if self.selected() == 0 {
            MENU_ITEMS.len() - 1
        } else {
            self.selected() - 1
        };
        self.list_state.select(Some(i));
    }
}

fn render(frame: &mut Frame, app: &mut App) {
    let [title_area, list_area, detail_area, status_area, help_area] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Fill(1),
        Constraint::Length(5),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    // Title
    let title = Paragraph::new(Line::from(vec![
        Span::styled(
            " ocrus ",
            Style::new()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" Lightning-fast Japanese OCR"),
    ]))
    .block(Block::bordered());
    frame.render_widget(title, title_area);

    // Menu list
    let items: Vec<ListItem> = MENU_ITEMS
        .iter()
        .map(|item| ListItem::new(item.label))
        .collect();

    let list = List::new(items)
        .block(Block::bordered().title("Operations"))
        .highlight_style(
            Style::new()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ");

    frame.render_stateful_widget(list, list_area, &mut app.list_state);

    // Detail panel
    let selected = &MENU_ITEMS[app.selected()];
    let detail = Paragraph::new(vec![
        Line::from(Span::styled(
            selected.label,
            Style::new().add_modifier(Modifier::BOLD),
        )),
        Line::raw(""),
        Line::raw(selected.description),
    ])
    .block(Block::bordered().title("Details"))
    .wrap(Wrap { trim: false });
    frame.render_widget(detail, detail_area);

    // Status
    let status_text = app
        .status
        .as_deref()
        .unwrap_or("Select an operation and press Enter to run.");
    let status = Paragraph::new(status_text).block(Block::bordered().title("Status"));
    frame.render_widget(status, status_area);

    // Help
    let help = Paragraph::new("j/k or Up/Down: navigate | Enter: run | q: quit");
    frame.render_widget(help, help_area);
}

fn run_command(paths: &Paths, idx: usize) -> Result<()> {
    let item = &MENU_ITEMS[idx];
    let args = (item.command_fn)(paths);
    if args.is_empty() {
        return Ok(());
    }

    let program = &args[0];
    let cmd_args = &args[1..];

    eprintln!("\n--- Running: {} {} ---\n", program, cmd_args.join(" "));

    let status = Command::new(program)
        .args(cmd_args)
        .current_dir(&paths.workspace_root)
        .status()?;

    if status.success() {
        eprintln!("\n--- Completed successfully ---");
    } else {
        eprintln!("\n--- Exited with: {} ---", status);
    }

    eprintln!("Press any key to return to TUI...");
    // Wait for a keypress
    loop {
        if event::poll(Duration::from_secs(60))?
            && let Event::Key(key) = event::read()?
            && key.kind == KeyEventKind::Press
        {
            break;
        }
    }

    Ok(())
}

pub fn run() -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();

    let model_dir = std::env::var("OCRUS_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/Users/owa"))
                .join(".ocrus/models")
        });

    let paths = Paths {
        workspace_root,
        model_dir,
    };

    let mut terminal = ratatui::init();
    let mut app = App::new();

    loop {
        terminal.draw(|frame| render(frame, &mut app))?;

        if event::poll(Duration::from_millis(100))?
            && let Event::Key(key) = event::read()?
        {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => break,
                KeyCode::Down | KeyCode::Char('j') => app.next(),
                KeyCode::Up | KeyCode::Char('k') => app.previous(),
                KeyCode::Enter => {
                    let idx = app.selected();
                    ratatui::restore();

                    match run_command(&paths, idx) {
                        Ok(()) => {
                            app.status = Some(format!("{} completed.", MENU_ITEMS[idx].label));
                        }
                        Err(e) => {
                            app.status = Some(format!("Error: {e}"));
                        }
                    }

                    terminal = ratatui::init();
                }
                _ => {}
            }
        }
    }

    ratatui::restore();
    Ok(())
}
