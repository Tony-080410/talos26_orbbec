use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::{self, Stdout},
    path::{Path, PathBuf},
    time::Duration,
};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

const RATINGS: [&str; 3] = ["Accuracy", "Left/Right Skew", "Up/Down Skew"];
const HISTORY_FILE: &str = "ratings.jsonl";
const SPACE_FILE: &str = "space.json";
const SUGGEST_AT_VISION: &str = "suggested.at_vision.toml";
const SUGGEST_HERO_SNIPPET: &str = "suggested.hero_snippet.txt";

const DEFAULT_DELAY_MIN: f64 = 0.0;
const DEFAULT_DELAY_MAX: f64 = 0.35;
const DEFAULT_CD_MIN: f64 = 0.25;
const DEFAULT_CD_MAX: f64 = 0.35;
const MUZZLE_RPY_DELTA_DEG: f64 = 3.0;
const MUZZLE_XYZ_DELTA_M: f64 = 0.02;

#[derive(Debug, Clone)]
struct PlannerConfig {
    source_path: PathBuf,
    entries: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Params {
    plan_to_muzzle_delay: f64,
    cd: f64,
    muzzle_rpy_rad: Vec3,
    muzzle_xyz_m: Vec3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Space {
    delay_min: f64,
    delay_max: f64,
    cd_min: f64,
    cd_max: f64,
    muzzle_rpy_center_rad: Vec3,
    muzzle_rpy_delta_rad: f64,
    muzzle_xyz_center_m: Vec3,
    muzzle_xyz_delta_m: f64,
}

#[derive(Debug, Clone)]
struct CurrentConfig {
    planner: PlannerConfig,
    params: Params,
    hero_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HistoryEntry {
    timestamp: String,
    accuracy: u8,
    left_right: u8,
    up_down: u8,
    score: u32,
    #[serde(default)]
    config: BTreeMap<String, String>,
    #[serde(default)]
    params: Option<Params>,
}

enum AppMode {
    Rate,
    Suggest,
}

struct App {
    config_path: PathBuf,
    hero_path: PathBuf,
    rateui_dir: PathBuf,
    history_path: PathBuf,
    space_path: PathBuf,
    suggest_at_vision_path: PathBuf,
    suggest_hero_snippet_path: PathBuf,
    space: Option<Space>,
    current: Option<CurrentConfig>,
    ratings: [Option<u8>; 3],
    selected: usize,
    suggestion: Option<SuggestionBundle>,
    error_message: Option<String>,
    mode: AppMode,
}

impl App {
    fn new(config_path: PathBuf, hero_path: PathBuf, rateui_dir: PathBuf) -> Self {
        let history_path = rateui_dir.join(HISTORY_FILE);
        let space_path = rateui_dir.join(SPACE_FILE);
        Self {
            config_path,
            hero_path,
            rateui_dir: rateui_dir.clone(),
            history_path,
            space_path,
            suggest_at_vision_path: rateui_dir.join(SUGGEST_AT_VISION),
            suggest_hero_snippet_path: rateui_dir.join(SUGGEST_HERO_SNIPPET),
            space: None,
            current: None,
            ratings: [None, None, None],
            selected: 0,
            suggestion: None,
            error_message: None,
            mode: AppMode::Rate,
        }
    }

    fn load_config(&mut self) {
        match read_current_config(&self.config_path, &self.hero_path) {
            Ok(current) => {
                self.error_message = None;
                self.current = Some(current);
            }
            Err(err) => {
                self.error_message = Some(format!("Failed to read config: {err}"));
                self.current = None;
                self.space = None;
            }
        }

        if self.space.is_none() {
            if let Some(current) = &self.current {
                match load_or_init_space(&self.space_path, &current.params) {
                    Ok(space) => self.space = Some(space),
                    Err(err) => {
                        self.error_message =
                            Some(format!("Failed to load/init optimizer space: {err}"));
                        self.space = None;
                    }
                }
            }
        }
    }

    fn reset_ratings(&mut self) {
        self.ratings = [None, None, None];
        self.selected = 0;
    }
}

#[derive(Debug, Clone)]
struct SuggestionBundle {
    next: Params,
    alternates: Vec<Params>,
    best_seen: Option<HistoryEntry>,
    saved_files: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let cwd = std::env::current_dir().context("get current dir")?;
    let config_path = find_config_path(&cwd);
    let hero_path = find_hero_path(&cwd);
    let rateui_dir = find_rateui_dir(&cwd)?;

    let mut terminal = setup_terminal()?;
    let result = run_app(&mut terminal, App::new(config_path, hero_path, rateui_dir));
    restore_terminal(&mut terminal)?;
    if let Err(err) = result {
        if err.to_string() == "exit" {
            return Ok(());
        }
        return Err(err);
    }
    Ok(())
}

fn find_config_path(cwd: &Path) -> PathBuf {
    let candidate = cwd.join("at_vision.toml");
    if candidate.exists() {
        return candidate;
    }
    if let Some(parent) = cwd.parent() {
        let parent_candidate = parent.join("at_vision.toml");
        if parent_candidate.exists() {
            return parent_candidate;
        }
    }
    candidate
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    Terminal::new(backend).context("create terminal")
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen).context("leave alternate screen")?;
    terminal.show_cursor().context("show cursor")?;
    Ok(())
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>, mut app: App) -> Result<()> {
    app.load_config();
    loop {
        terminal
            .draw(|frame| draw_ui(frame, &app))
            .context("draw ui")?;

        if event::poll(Duration::from_millis(200)).context("poll events")? {
            if let Event::Key(key) = event::read().context("read event")? {
                if key.kind == KeyEventKind::Press {
                    let should_exit = match app.mode {
                        AppMode::Rate => handle_rate_keys(&mut app, key.code)?,
                        AppMode::Suggest => handle_suggest_keys(&mut app, key.code)?,
                    };
                    if should_exit {
                        return Ok(());
                    }
                }
            }
        }
    }
}

fn handle_rate_keys(app: &mut App, code: KeyCode) -> Result<bool> {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            return Err(anyhow::anyhow!("exit"));
        }
        KeyCode::Up => {
            if app.selected > 0 {
                app.selected -= 1;
            }
        }
        KeyCode::Down => {
            if app.selected + 1 < RATINGS.len() {
                app.selected += 1;
            }
        }
        KeyCode::Char('r') => {
            app.space = None;
            app.load_config();
        }
        KeyCode::Char('1')
        | KeyCode::Char('2')
        | KeyCode::Char('3')
        | KeyCode::Char('4')
        | KeyCode::Char('5') => {
            let value = match code {
                KeyCode::Char('1') => 1,
                KeyCode::Char('2') => 2,
                KeyCode::Char('3') => 3,
                KeyCode::Char('4') => 4,
                KeyCode::Char('5') => 5,
                _ => 1,
            };
            app.ratings[app.selected] = Some(value);
        }
        KeyCode::Enter => {
            if app.ratings.iter().all(|value| value.is_some()) {
                let Some(current) = &app.current else {
                    app.error_message = Some("No config loaded; cannot save rating.".to_string());
                    return Ok(false);
                };
                let entry = HistoryEntry {
                    timestamp: OffsetDateTime::now_utc()
                        .format(&Rfc3339)
                        .unwrap_or_else(|_| "unknown".to_string()),
                    accuracy: app.ratings[0].unwrap_or(0),
                    left_right: app.ratings[1].unwrap_or(0),
                    up_down: app.ratings[2].unwrap_or(0),
                    score: score_entry(&app.ratings),
                    config: current.planner.entries.clone(),
                    params: Some(current.params.clone()),
                };
                save_history(&app.history_path, &entry)?;
                app.suggestion = build_suggestion_bundle(app)?;
                app.mode = AppMode::Suggest;
            }
        }
        _ => {}
    }
    Ok(false)
}

fn handle_suggest_keys(app: &mut App, code: KeyCode) -> Result<bool> {
    match code {
        KeyCode::Char('q') | KeyCode::Esc => {
            return Err(anyhow::anyhow!("exit"));
        }
        KeyCode::Enter | KeyCode::Char('n') => {
            app.reset_ratings();
            app.load_config();
            app.mode = AppMode::Rate;
        }
        _ => {}
    }
    Ok(false)
}

fn draw_ui(frame: &mut ratatui::Frame, app: &App) {
    let size = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(5),
            Constraint::Length(3),
        ])
        .split(size);

    let header = draw_header(app, chunks[0]);
    frame.render_widget(header, chunks[0]);

    let body_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(chunks[1]);

    let left = draw_left_panel(app, body_chunks[0]);
    frame.render_widget(left, body_chunks[0]);

    let right = draw_right_panel(app, body_chunks[1]);
    frame.render_widget(right, body_chunks[1]);

    let footer = draw_footer(app, chunks[2]);
    frame.render_widget(footer, chunks[2]);
}

fn draw_header(app: &App, _area: Rect) -> Paragraph<'static> {
    let mut lines = vec![Line::from(Span::styled(
        "RateUI Planner Scorer",
        Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD),
    ))];
    lines.push(Line::from(format!(
        "at_vision: {}",
        app.config_path.display()
    )));
    lines.push(Line::from(format!("hero.hpp: {}", app.hero_path.display())));
    if let Some(space) = &app.space {
        lines.push(Line::from(format!(
            "Space: delay[{:.2},{:.2}] cd[{:.2},{:.2}] rpy±{:.1}deg xyz±{:.0}mm",
            space.delay_min,
            space.delay_max,
            space.cd_min,
            space.cd_max,
            rad_to_deg(space.muzzle_rpy_delta_rad),
            space.muzzle_xyz_delta_m * 1000.0
        )));
    }
    if let Some(err) = &app.error_message {
        lines.push(Line::from(Span::styled(
            err.clone(),
            Style::default().fg(Color::Red),
        )));
    }
    Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Status"))
}

fn draw_left_panel(app: &App, _area: Rect) -> Paragraph<'static> {
    let mut lines = Vec::new();
    match app.mode {
        AppMode::Rate => {
            if let Some(current) = &app.current {
                lines.push(Line::from("Current parameters to rate:"));
                lines.push(Line::from(""));
                lines.extend(format_params_lines(&current.params));
            } else {
                lines.push(Line::from("No planner config loaded."));
                lines.push(Line::from("Press r to retry."));
            }
        }
        AppMode::Suggest => {
            if let Some(bundle) = &app.suggestion {
                lines.push(Line::from("Next suggested parameters:"));
                lines.push(Line::from(""));
                lines.extend(format_params_lines(&bundle.next));
                lines.push(Line::from(""));
                if !bundle.alternates.is_empty() {
                    lines.push(Line::from("Alternates:"));
                    for (idx, alt) in bundle.alternates.iter().enumerate() {
                        lines.push(Line::from(format!("Alt {}:", idx + 1)));
                        lines.extend(format_params_lines_compact(alt));
                        lines.push(Line::from(""));
                    }
                }
                if let Some(best) = &bundle.best_seen {
                    lines.push(Line::from(format!(
                        "Best seen score: {} | {}",
                        best.score, best.timestamp
                    )));
                }
                if !bundle.saved_files.is_empty() {
                    lines.push(Line::from("Saved:"));
                    for path in &bundle.saved_files {
                        lines.push(Line::from(format!("- {}", path.display())));
                    }
                }
            } else {
                lines.push(Line::from("No history yet."));
            }
        }
    }

    Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Config"))
}

fn draw_right_panel(app: &App, _area: Rect) -> Paragraph<'static> {
    let lines = match app.mode {
        AppMode::Rate => rating_lines(&app.ratings, app.selected),
        AppMode::Suggest => suggestion_lines(app),
    };
    Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Rating"))
}

fn draw_footer(app: &App, _area: Rect) -> Paragraph<'static> {
    let lines = match app.mode {
        AppMode::Rate => vec![
            Line::from("Up/Down select field, 1-5 to set score"),
            Line::from("Enter to save + suggest next, r to reload, q to quit"),
        ],
        AppMode::Suggest => vec![
            Line::from("Apply suggested files manually, then Enter for next rating"),
            Line::from("q to quit"),
        ],
    };
    Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title("Help"))
}

fn rating_lines(ratings: &[Option<u8>; 3], selected: usize) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    for (idx, label) in RATINGS.iter().enumerate() {
        let marker = if idx == selected { ">" } else { " " };
        let value = ratings[idx]
            .map(|v| format!("{}/5", v))
            .unwrap_or_else(|| "[ ]".to_string());
        lines.push(Line::from(format!("{marker} {label}: {value}")));
    }
    lines.push(Line::from(""));
    lines.push(Line::from("All fields required to submit."));
    lines
}

fn suggestion_lines(app: &App) -> Vec<Line<'static>> {
    if let Some(bundle) = &app.suggestion {
        let Some(best) = &bundle.best_seen else {
            return vec![Line::from("No saved rating yet.")];
        };
        vec![
            Line::from("Best rating summary:"),
            Line::from(format!("Accuracy: {}/5", best.accuracy)),
            Line::from(format!("Left/Right: {}/5", best.left_right)),
            Line::from(format!("Up/Down: {}/5", best.up_down)),
            Line::from(format!("Score: {}", best.score)),
        ]
    } else {
        vec![Line::from("No saved rating yet.")]
    }
}

fn format_config_lines(entries: &BTreeMap<String, String>) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if entries.is_empty() {
        lines.push(Line::from("No planner settings found."));
        return lines;
    }
    for (key, value) in entries {
        lines.push(Line::from(format!("{key} = {value}")));
    }
    lines
}

fn read_planner_config(path: &Path) -> Result<BTreeMap<String, String>> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: toml::Value = toml::from_str(&raw).context("parse toml")?;
    let planner = value
        .get("planner")
        .and_then(|v| v.as_table())
        .context("missing [planner] table")?;
    let mut entries = BTreeMap::new();
    flatten_table("", planner, &mut entries);
    Ok(entries)
}

fn read_planner_params(path: &Path) -> Result<(f64, f64)> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: toml::Value = toml::from_str(&raw).context("parse toml")?;
    let planner = value
        .get("planner")
        .and_then(|v| v.as_table())
        .context("missing [planner] table")?;
    let delay = planner
        .get("plan_to_muzzle_delay")
        .and_then(toml_number_to_f64)
        .context("missing planner.plan_to_muzzle_delay")?;

    let cd = planner
        .get("trajectory")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("cd"))
        .and_then(toml_number_to_f64)
        .context("missing planner.trajectory.cd")?;

    Ok((delay, cd))
}

fn toml_number_to_f64(v: &toml::Value) -> Option<f64> {
    v.as_float()
        .or_else(|| v.as_integer().map(|i| i as f64))
        .or_else(|| v.as_str().and_then(|s| s.parse::<f64>().ok()))
}

fn read_current_config(config_path: &Path, hero_path: &Path) -> Result<CurrentConfig> {
    let entries = read_planner_config(config_path)?;
    let (delay, cd) = read_planner_params(config_path)?;
    let muzzle = read_hero_muzzle_transform(hero_path)?;

    Ok(CurrentConfig {
        planner: PlannerConfig {
            source_path: config_path.to_path_buf(),
            entries,
        },
        params: Params {
            plan_to_muzzle_delay: delay,
            cd,
            muzzle_rpy_rad: muzzle.rpy_rad,
            muzzle_xyz_m: muzzle.xyz_m,
        },
        hero_path: hero_path.to_path_buf(),
    })
}

fn flatten_table(prefix: &str, table: &toml::value::Table, out: &mut BTreeMap<String, String>) {
    for (key, value) in table {
        match value {
            toml::Value::Table(inner) => {
                let next_prefix = format!("{prefix}{key}.");
                flatten_table(&next_prefix, inner, out);
            }
            toml::Value::String(text) => {
                out.insert(format!("{prefix}{key}"), text.clone());
            }
            _ => {
                out.insert(format!("{prefix}{key}"), value.to_string());
            }
        }
    }
}

fn score_entry(ratings: &[Option<u8>; 3]) -> u32 {
    let accuracy = ratings[0].unwrap_or(0) as u32;
    let left_right = ratings[1].unwrap_or(0) as u32;
    let up_down = ratings[2].unwrap_or(0) as u32;
    accuracy * 2 + left_right + up_down
}

fn save_history(path: &Path, entry: &HistoryEntry) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let line = serde_json::to_string(entry).context("serialize entry")?;
    fs::write(path, append_line(path, &line)).context("write history")?;
    Ok(())
}

fn append_line(path: &Path, line: &str) -> String {
    if let Ok(existing) = fs::read_to_string(path) {
        if existing.ends_with('\n') || existing.is_empty() {
            format!("{existing}{line}\n")
        } else {
            format!("{existing}\n{line}\n")
        }
    } else {
        format!("{line}\n")
    }
}

fn load_best_history(path: &Path) -> Result<Option<HistoryEntry>> {
    let raw = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return Ok(None),
    };
    let mut best: Option<HistoryEntry> = None;
    let mut best_index = 0usize;
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<HistoryEntry>(line) {
            let replace = match &best {
                None => true,
                Some(current) => {
                    entry.score > current.score
                        || (entry.score == current.score && idx > best_index)
                }
            };
            if replace {
                best = Some(entry);
                best_index = idx;
            }
        }
    }
    Ok(best)
}

fn find_rateui_dir(cwd: &Path) -> Result<PathBuf> {
    let dir = cwd.join(".rateui");
    fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    Ok(dir)
}

fn find_hero_path(cwd: &Path) -> PathBuf {
    let candidate = cwd
        .join("include")
        .join("math")
        .join("fast_tf")
        .join("hero.hpp");
    if candidate.exists() {
        return candidate;
    }
    if let Some(parent) = cwd.parent() {
        let parent_candidate = parent
            .join("include")
            .join("math")
            .join("fast_tf")
            .join("hero.hpp");
        if parent_candidate.exists() {
            return parent_candidate;
        }
    }
    candidate
}

fn deg_to_rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / std::f64::consts::PI
}

fn load_or_init_space(space_path: &Path, current_params: &Params) -> Result<Space> {
    if let Ok(raw) = fs::read_to_string(space_path) {
        let space: Space = serde_json::from_str(&raw).context("parse space.json")?;
        return Ok(space);
    }
    let space = Space {
        delay_min: DEFAULT_DELAY_MIN,
        delay_max: DEFAULT_DELAY_MAX,
        cd_min: DEFAULT_CD_MIN,
        cd_max: DEFAULT_CD_MAX,
        muzzle_rpy_center_rad: current_params.muzzle_rpy_rad,
        muzzle_rpy_delta_rad: deg_to_rad(MUZZLE_RPY_DELTA_DEG),
        muzzle_xyz_center_m: current_params.muzzle_xyz_m,
        muzzle_xyz_delta_m: MUZZLE_XYZ_DELTA_M,
    };
    fs::write(
        space_path,
        serde_json::to_string_pretty(&space).context("serialize space.json")?,
    )
    .with_context(|| format!("write {}", space_path.display()))?;
    Ok(space)
}

#[derive(Debug, Clone, Copy)]
struct HeroMuzzleTransform {
    rpy_rad: Vec3,
    xyz_m: Vec3,
}

fn read_hero_muzzle_transform(path: &Path) -> Result<HeroMuzzleTransform> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let anchor = "update<muzzle_link_fuxk_frame>";
    let anchor_idx = raw
        .find(anchor)
        .context("missing muzzle_link_fuxk_frame update")?;
    let slice = &raw[anchor_idx..];
    let call = "TransformMatrix4d::from_rpy(";
    let call_idx = slice
        .find(call)
        .context("missing TransformMatrix4d::from_rpy(...)")?;
    let args_start = anchor_idx + call_idx + call.len();
    let args_end = raw[args_start..]
        .find(')')
        .map(|off| args_start + off)
        .context("unterminated from_rpy call")?;
    let args = &raw[args_start..args_end];
    let parts: Vec<&str> = args
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() != 6 {
        anyhow::bail!("expected 6 args in muzzle from_rpy, got {}", parts.len());
    }
    let values: Vec<f64> = parts
        .iter()
        .map(|s| parse_expr_f64(s))
        .collect::<Result<Vec<_>>>()?;
    Ok(HeroMuzzleTransform {
        rpy_rad: Vec3::new(values[0], values[1], values[2]),
        xyz_m: Vec3::new(values[3], values[4], values[5]),
    })
}

fn parse_expr_f64(token: &str) -> Result<f64> {
    let cleaned = token.replace(' ', "");
    if let Ok(v) = cleaned.parse::<f64>() {
        return Ok(v);
    }
    let pi = std::f64::consts::PI;
    if cleaned == "pi" {
        return Ok(pi);
    }
    if cleaned == "-pi" {
        return Ok(-pi);
    }
    if let Some(rest) = cleaned.strip_prefix("pi/") {
        let denom: f64 = rest.parse().context("parse pi/denom")?;
        return Ok(pi / denom);
    }
    if let Some(rest) = cleaned.strip_prefix("-pi/") {
        let denom: f64 = rest.parse().context("parse -pi/denom")?;
        return Ok(-pi / denom);
    }
    anyhow::bail!("unsupported numeric expression: {token}")
}

fn format_params_lines(params: &Params) -> Vec<Line<'static>> {
    vec![
        Line::from(format!(
            "plan_to_muzzle_delay = {:.4} s",
            params.plan_to_muzzle_delay
        )),
        Line::from(format!("cd = {:.4}", params.cd)),
        Line::from(format!(
            "muzzle rpy = [{:.3}, {:.3}, {:.3}] deg",
            rad_to_deg(params.muzzle_rpy_rad.x),
            rad_to_deg(params.muzzle_rpy_rad.y),
            rad_to_deg(params.muzzle_rpy_rad.z)
        )),
        Line::from(format!(
            "muzzle xyz = [{:.4}, {:.4}, {:.4}] m",
            params.muzzle_xyz_m.x, params.muzzle_xyz_m.y, params.muzzle_xyz_m.z
        )),
    ]
}

fn format_params_lines_compact(params: &Params) -> Vec<Line<'static>> {
    vec![Line::from(format!(
        "delay {:.4}s | cd {:.4} | rpy [{:.2},{:.2},{:.2}]deg | xyz [{:.3},{:.3},{:.3}]m",
        params.plan_to_muzzle_delay,
        params.cd,
        rad_to_deg(params.muzzle_rpy_rad.x),
        rad_to_deg(params.muzzle_rpy_rad.y),
        rad_to_deg(params.muzzle_rpy_rad.z),
        params.muzzle_xyz_m.x,
        params.muzzle_xyz_m.y,
        params.muzzle_xyz_m.z
    ))]
}

fn build_suggestion_bundle(app: &mut App) -> Result<Option<SuggestionBundle>> {
    let Some(space) = &app.space else {
        return Ok(None);
    };
    let Some(current) = &app.current else {
        return Ok(None);
    };
    let trials = load_trials(&app.history_path)?;
    let best_seen = load_best_history(&app.history_path)?;
    let mut rng = Rng::new(seed_from_time());
    let suggestions = suggest_next(space, &trials, &mut rng);
    let Some(next) = suggestions.first().cloned() else {
        return Ok(None);
    };
    let alternates = suggestions.into_iter().skip(1).collect::<Vec<_>>();

    let mut saved_files = Vec::new();
    saved_files.extend(save_suggested_files(
        &app.suggest_at_vision_path,
        &app.suggest_hero_snippet_path,
        &next,
        &current.hero_path,
    )?);

    Ok(Some(SuggestionBundle {
        next,
        alternates,
        best_seen,
        saved_files,
    }))
}

fn save_suggested_files(
    at_vision_out: &Path,
    hero_snippet_out: &Path,
    params: &Params,
    hero_path: &Path,
) -> Result<Vec<PathBuf>> {
    let mut written = Vec::new();

    let at_vision = format!(
        "[planner]\nplan_to_muzzle_delay = {:.6}\n\n[planner.trajectory]\ncd = {:.6}\n",
        params.plan_to_muzzle_delay, params.cd
    );
    fs::write(at_vision_out, at_vision)
        .with_context(|| format!("write {}", at_vision_out.display()))?;
    written.push(at_vision_out.to_path_buf());

    let snippet = format!(
        "// Apply to {}\nTransformMatrix4d::from_rpy({:.6}, {:.6}, {:.6}, {:.6}, {:.6}, {:.6})\n",
        hero_path.display(),
        params.muzzle_rpy_rad.x,
        params.muzzle_rpy_rad.y,
        params.muzzle_rpy_rad.z,
        params.muzzle_xyz_m.x,
        params.muzzle_xyz_m.y,
        params.muzzle_xyz_m.z
    );
    fs::write(hero_snippet_out, snippet)
        .with_context(|| format!("write {}", hero_snippet_out.display()))?;
    written.push(hero_snippet_out.to_path_buf());

    Ok(written)
}

#[derive(Debug, Clone)]
struct Trial {
    params: Params,
    score: f64,
}

fn load_trials(path: &Path) -> Result<Vec<Trial>> {
    let raw = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(_) => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    for line in raw.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(entry) = serde_json::from_str::<HistoryEntry>(line) else {
            continue;
        };
        let Some(params) = entry.params else {
            continue;
        };
        out.push(Trial {
            params,
            score: entry.score as f64,
        });
    }
    Ok(out)
}

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 { 0x9e3779b97f4a7c15 } else { seed };
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        let u = self.next_u64() >> 11;
        (u as f64) / ((1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + z0 * std
    }
}

fn seed_from_time() -> u64 {
    let now = OffsetDateTime::now_utc().unix_timestamp_nanos();
    now as u64 ^ ((now >> 32) as u64).wrapping_mul(0x9e3779b97f4a7c15)
}

fn suggest_next(space: &Space, trials: &[Trial], rng: &mut Rng) -> Vec<Params> {
    if trials.is_empty() {
        return (0..3).map(|_| random_uniform_params(space, rng)).collect();
    }
    if trials.len() < 8 {
        return bootstrapped_suggestions(space, trials, rng, 3);
    }
    tpe_suggestions(space, trials, rng, 3)
}

fn bootstrapped_suggestions(
    space: &Space,
    trials: &[Trial],
    rng: &mut Rng,
    k: usize,
) -> Vec<Params> {
    let mut out = Vec::new();
    let best = trials
        .iter()
        .max_by(|a, b| a.score.total_cmp(&b.score))
        .map(|t| t.params.clone())
        .unwrap_or_else(|| random_uniform_params(space, rng));

    out.push(local_perturb(space, rng, &best));
    let rest = space_filling_suggestions(space, trials, rng, k.saturating_sub(1));
    out.extend(rest);
    out.truncate(k);
    out
}

fn space_filling_suggestions(
    space: &Space,
    trials: &[Trial],
    rng: &mut Rng,
    k: usize,
) -> Vec<Params> {
    let mut suggestions = Vec::new();
    for _ in 0..k {
        let mut best: Option<(Params, f64)> = None;
        for _ in 0..300 {
            let cand = random_uniform_params(space, rng);
            let d = min_normalized_distance(space, trials, &suggestions, &cand);
            if best.as_ref().map(|(_, bd)| d > *bd).unwrap_or(true) {
                best = Some((cand, d));
            }
        }
        if let Some((p, _)) = best {
            suggestions.push(p);
        }
    }
    suggestions
}

fn local_perturb(space: &Space, rng: &mut Rng, base: &Params) -> Params {
    let clamp = |v: f64, lo: f64, hi: f64| v.clamp(lo, hi);
    let delay_std = 0.08 * (space.delay_max - space.delay_min);
    let cd_std = 0.12 * (space.cd_max - space.cd_min);
    let rpy_std = 0.35 * space.muzzle_rpy_delta_rad;
    let xyz_std = 0.35 * space.muzzle_xyz_delta_m;

    Params {
        plan_to_muzzle_delay: clamp(
            rng.normal(base.plan_to_muzzle_delay, delay_std),
            space.delay_min,
            space.delay_max,
        ),
        cd: clamp(rng.normal(base.cd, cd_std), space.cd_min, space.cd_max),
        muzzle_rpy_rad: Vec3::new(
            clamp(
                rng.normal(base.muzzle_rpy_rad.x, rpy_std),
                space.muzzle_rpy_center_rad.x - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.x + space.muzzle_rpy_delta_rad,
            ),
            clamp(
                rng.normal(base.muzzle_rpy_rad.y, rpy_std),
                space.muzzle_rpy_center_rad.y - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.y + space.muzzle_rpy_delta_rad,
            ),
            clamp(
                rng.normal(base.muzzle_rpy_rad.z, rpy_std),
                space.muzzle_rpy_center_rad.z - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.z + space.muzzle_rpy_delta_rad,
            ),
        ),
        muzzle_xyz_m: Vec3::new(
            clamp(
                rng.normal(base.muzzle_xyz_m.x, xyz_std),
                space.muzzle_xyz_center_m.x - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.x + space.muzzle_xyz_delta_m,
            ),
            clamp(
                rng.normal(base.muzzle_xyz_m.y, xyz_std),
                space.muzzle_xyz_center_m.y - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.y + space.muzzle_xyz_delta_m,
            ),
            clamp(
                rng.normal(base.muzzle_xyz_m.z, xyz_std),
                space.muzzle_xyz_center_m.z - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.z + space.muzzle_xyz_delta_m,
            ),
        ),
    }
}

fn tpe_suggestions(space: &Space, trials: &[Trial], rng: &mut Rng, k: usize) -> Vec<Params> {
    let mut sorted = trials.to_vec();
    sorted.sort_by(|a, b| b.score.total_cmp(&a.score));
    let n = sorted.len();
    let n_good = ((n as f64) * 0.2).ceil().clamp(5.0, (n - 1) as f64) as usize;
    let (good, bad) = sorted.split_at(n_good);

    let mut candidates: Vec<(Params, f64)> = Vec::new();
    for _ in 0..800 {
        let explore = rng.next_f64() < 0.25;
        let cand = if explore {
            random_uniform_params(space, rng)
        } else {
            sample_from_good(space, good, rng)
        };
        let ratio = log_density_ratio(space, good, bad, &cand);
        let d = min_normalized_distance(space, trials, &[], &cand);
        let score = ratio + 0.1 * d;
        candidates.push((cand, score));
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut out = Vec::new();
    for (cand, _) in candidates {
        if out.len() >= k {
            break;
        }
        let d = min_normalized_distance(space, trials, &out, &cand);
        if d > 0.08 {
            out.push(cand);
        }
    }
    if out.len() < k {
        while out.len() < k {
            out.push(random_uniform_params(space, rng));
        }
    }
    out
}

fn random_uniform_params(space: &Space, rng: &mut Rng) -> Params {
    Params {
        plan_to_muzzle_delay: rng.uniform(space.delay_min, space.delay_max),
        cd: rng.uniform(space.cd_min, space.cd_max),
        muzzle_rpy_rad: Vec3::new(
            rng.uniform(
                space.muzzle_rpy_center_rad.x - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.x + space.muzzle_rpy_delta_rad,
            ),
            rng.uniform(
                space.muzzle_rpy_center_rad.y - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.y + space.muzzle_rpy_delta_rad,
            ),
            rng.uniform(
                space.muzzle_rpy_center_rad.z - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.z + space.muzzle_rpy_delta_rad,
            ),
        ),
        muzzle_xyz_m: Vec3::new(
            rng.uniform(
                space.muzzle_xyz_center_m.x - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.x + space.muzzle_xyz_delta_m,
            ),
            rng.uniform(
                space.muzzle_xyz_center_m.y - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.y + space.muzzle_xyz_delta_m,
            ),
            rng.uniform(
                space.muzzle_xyz_center_m.z - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.z + space.muzzle_xyz_delta_m,
            ),
        ),
    }
}

fn sample_from_good(space: &Space, good: &[Trial], rng: &mut Rng) -> Params {
    let mut delays = Vec::new();
    let mut cds = Vec::new();
    let mut rxs = Vec::new();
    let mut rys = Vec::new();
    let mut rzs = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for t in good {
        delays.push(t.params.plan_to_muzzle_delay);
        cds.push(t.params.cd);
        rxs.push(t.params.muzzle_rpy_rad.x);
        rys.push(t.params.muzzle_rpy_rad.y);
        rzs.push(t.params.muzzle_rpy_rad.z);
        xs.push(t.params.muzzle_xyz_m.x);
        ys.push(t.params.muzzle_xyz_m.y);
        zs.push(t.params.muzzle_xyz_m.z);
    }

    let bandwidth = |vals: &[f64]| {
        if vals.len() < 2 {
            return 1e-6;
        }
        let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (vals.len() as f64);
        let std = var.sqrt().max(1e-9);
        (1.06 * std * (vals.len() as f64).powf(-0.2)).max(1e-6)
    };
    let mut sample_1d = |vals: &Vec<f64>| {
        let mu = pick_random(rng, vals);
        let std = bandwidth(vals);
        rng.normal(mu, std)
    };

    let clamp = |v: f64, lo: f64, hi: f64| v.clamp(lo, hi);

    Params {
        plan_to_muzzle_delay: clamp(sample_1d(&delays), space.delay_min, space.delay_max),
        cd: clamp(sample_1d(&cds), space.cd_min, space.cd_max),
        muzzle_rpy_rad: Vec3::new(
            clamp(
                sample_1d(&rxs),
                space.muzzle_rpy_center_rad.x - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.x + space.muzzle_rpy_delta_rad,
            ),
            clamp(
                sample_1d(&rys),
                space.muzzle_rpy_center_rad.y - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.y + space.muzzle_rpy_delta_rad,
            ),
            clamp(
                sample_1d(&rzs),
                space.muzzle_rpy_center_rad.z - space.muzzle_rpy_delta_rad,
                space.muzzle_rpy_center_rad.z + space.muzzle_rpy_delta_rad,
            ),
        ),
        muzzle_xyz_m: Vec3::new(
            clamp(
                sample_1d(&xs),
                space.muzzle_xyz_center_m.x - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.x + space.muzzle_xyz_delta_m,
            ),
            clamp(
                sample_1d(&ys),
                space.muzzle_xyz_center_m.y - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.y + space.muzzle_xyz_delta_m,
            ),
            clamp(
                sample_1d(&zs),
                space.muzzle_xyz_center_m.z - space.muzzle_xyz_delta_m,
                space.muzzle_xyz_center_m.z + space.muzzle_xyz_delta_m,
            ),
        ),
    }
}

fn pick_random(rng: &mut Rng, vals: &[f64]) -> f64 {
    let idx = (rng.next_f64() * (vals.len() as f64)).floor() as usize;
    vals[idx.min(vals.len().saturating_sub(1))]
}

fn log_density_ratio(space: &Space, good: &[Trial], bad: &[Trial], params: &Params) -> f64 {
    let log_good = log_marginal(space, good, params, true);
    let log_bad = log_marginal(space, bad, params, false);
    log_good - log_bad
}

fn log_marginal(_space: &Space, trials: &[Trial], params: &Params, _is_good: bool) -> f64 {
    if trials.is_empty() {
        return -1e9;
    }
    let dims = [
        params.plan_to_muzzle_delay,
        params.cd,
        params.muzzle_rpy_rad.x,
        params.muzzle_rpy_rad.y,
        params.muzzle_rpy_rad.z,
        params.muzzle_xyz_m.x,
        params.muzzle_xyz_m.y,
        params.muzzle_xyz_m.z,
    ];
    let mut total = 0.0;
    for (dim_idx, x) in dims.into_iter().enumerate() {
        let vals: Vec<f64> = trials
            .iter()
            .map(|t| match dim_idx {
                0 => t.params.plan_to_muzzle_delay,
                1 => t.params.cd,
                2 => t.params.muzzle_rpy_rad.x,
                3 => t.params.muzzle_rpy_rad.y,
                4 => t.params.muzzle_rpy_rad.z,
                5 => t.params.muzzle_xyz_m.x,
                6 => t.params.muzzle_xyz_m.y,
                _ => t.params.muzzle_xyz_m.z,
            })
            .collect();
        let h = silverman_bandwidth(&vals);
        total += log_gaussian_mixture(x, &vals, h);
    }
    total
}

fn silverman_bandwidth(vals: &[f64]) -> f64 {
    if vals.len() < 2 {
        return 1e-6;
    }
    let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (vals.len() as f64);
    let std = var.sqrt().max(1e-9);
    (1.06 * std * (vals.len() as f64).powf(-0.2)).max(1e-6)
}

fn log_gaussian_mixture(x: f64, centers: &[f64], std: f64) -> f64 {
    let var = std * std;
    let norm = -0.5 * (2.0 * std::f64::consts::PI * var).ln();
    let mut max_log = f64::NEG_INFINITY;
    let mut logs = Vec::with_capacity(centers.len());
    for &c in centers {
        let z = (x - c) * (x - c) / var;
        let lp = norm - 0.5 * z;
        if lp > max_log {
            max_log = lp;
        }
        logs.push(lp);
    }
    let sum_exp = logs
        .iter()
        .map(|lp| (*lp - max_log).exp())
        .sum::<f64>()
        .max(1e-300);
    (sum_exp / (centers.len() as f64)).ln() + max_log
}

fn min_normalized_distance(
    space: &Space,
    trials: &[Trial],
    extra: &[Params],
    params: &Params,
) -> f64 {
    let mut best = f64::INFINITY;
    for t in trials {
        best = best.min(normalized_l2(space, &t.params, params));
    }
    for p in extra {
        best = best.min(normalized_l2(space, p, params));
    }
    if best.is_infinite() {
        return 1.0;
    }
    best
}

fn normalized_l2(space: &Space, a: &Params, b: &Params) -> f64 {
    let mut acc = 0.0;
    acc +=
        sq((a.plan_to_muzzle_delay - b.plan_to_muzzle_delay) / (space.delay_max - space.delay_min));
    acc += sq((a.cd - b.cd) / (space.cd_max - space.cd_min));

    let r = space.muzzle_rpy_delta_rad.max(1e-9);
    acc += sq((a.muzzle_rpy_rad.x - b.muzzle_rpy_rad.x) / (2.0 * r));
    acc += sq((a.muzzle_rpy_rad.y - b.muzzle_rpy_rad.y) / (2.0 * r));
    acc += sq((a.muzzle_rpy_rad.z - b.muzzle_rpy_rad.z) / (2.0 * r));

    let t = space.muzzle_xyz_delta_m.max(1e-9);
    acc += sq((a.muzzle_xyz_m.x - b.muzzle_xyz_m.x) / (2.0 * t));
    acc += sq((a.muzzle_xyz_m.y - b.muzzle_xyz_m.y) / (2.0 * t));
    acc += sq((a.muzzle_xyz_m.z - b.muzzle_xyz_m.z) / (2.0 * t));

    acc.sqrt()
}

fn sq(x: f64) -> f64 {
    x * x
}
