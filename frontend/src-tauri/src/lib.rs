use std::io::Write;
use std::process::{Child, Command};
use std::sync::Mutex;
use tauri::Manager;

static FLASK_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

// ── Logging ────────────────────────────────────────────────────────────────

fn get_log_path() -> std::path::PathBuf {
    // temp_dir() is ALWAYS writable on every OS (Windows, macOS, Linux)
    std::env::temp_dir().join("hatsun_startup.log")
}

fn log(msg: &str) {
    let path = get_log_path();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let _ = writeln!(f, "[{}] {}", ts, msg);
    }
    // Also print to stderr so it shows in any attached console
    eprintln!("[HATSUN] {}", msg);
}

// ── Writable data dir ──────────────────────────────────────────────────────

fn get_data_dir(app: &tauri::App) -> std::path::PathBuf {
    // Try Tauri's API first
    if let Ok(d) = app.path().app_local_data_dir() {
        log(&format!("data_dir (tauri API): {:?}", d));
        return d;
    }

    // Windows fallback
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        if !local.is_empty() {
            let d = std::path::PathBuf::from(local).join("HatsunVRP");
            log(&format!("data_dir (LOCALAPPDATA): {:?}", d));
            return d;
        }
    }

    // macOS fallback: ~/Library/Application Support/HatsunVRP
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            let d = std::path::PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("HatsunVRP");
            log(&format!("data_dir (macOS HOME): {:?}", d));
            return d;
        }
    }

    // Last resort — use temp dir (always writable)
    let d = std::env::temp_dir().join("HatsunVRP");
    log(&format!("data_dir (temp fallback): {:?}", d));
    d
}

// ── Python finder ──────────────────────────────────────────────────────────

fn find_python() -> Option<String> {
    let mut candidates: Vec<String> = vec![
        "python".into(),
        "python3".into(),
        "python.exe".into(),
        "python3.exe".into(),
        // Common Windows system-wide paths
        r"C:\Python312\python.exe".into(),
        r"C:\Python311\python.exe".into(),
        r"C:\Python310\python.exe".into(),
        r"C:\Python39\python.exe".into(),
        // macOS Homebrew / system
        "/usr/bin/python3".into(),
        "/usr/local/bin/python3".into(),
        "/opt/homebrew/bin/python3".into(),
    ];

    // Add per-user AppData paths on Windows
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let root = std::path::Path::new(&local).join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                candidates.push(
                    entry.path().join("python.exe").to_string_lossy().to_string(),
                );
            }
        }
    }

    // Also check USERPROFILE\AppData\Local\Programs\Python on Windows
    if let Ok(profile) = std::env::var("USERPROFILE") {
        let root = std::path::Path::new(&profile)
            .join("AppData").join("Local").join("Programs").join("Python");
        if let Ok(entries) = std::fs::read_dir(&root) {
            for entry in entries.flatten() {
                candidates.push(
                    entry.path().join("python.exe").to_string_lossy().to_string(),
                );
            }
        }
    }

    log(&format!("searching {} python candidates", candidates.len()));
    for candidate in &candidates {
        match Command::new(candidate).args(["--version"]).output() {
            Ok(out) if out.status.success() => {
                let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
                log(&format!("  FOUND: {} -> {}", candidate, ver));
                return Some(candidate.clone());
            }
            Ok(out) => {
                log(&format!("  bad exit: {} ({})", candidate, String::from_utf8_lossy(&out.stderr).trim()));
            }
            Err(e) => {
                log(&format!("  not found: {} -> {}", candidate, e));
            }
        }
    }

    None
}

// ── Tauri app ──────────────────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .setup(|app| {
            // Reset log file (so each run is fresh)
            let _ = std::fs::write(get_log_path(), "");
            log("=== HatsunVRP startup ===");
            log(&format!("log file: {:?}", get_log_path()));

            // ── Script dir ─────────────────────────────────────────────────
            let script_dir = if cfg!(debug_assertions) {
                let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                manifest_dir.join("..").join("..").join("script")
            } else {
                app.path().resource_dir().unwrap().join("script")
            };
            let server_path = script_dir.join("server.py");

            log(&format!("script_dir  = {:?}", script_dir));
            log(&format!("server_path = {:?}", server_path));
            log(&format!("script_dir exists  = {}", script_dir.exists()));
            log(&format!("server.py  exists  = {}", server_path.exists()));

            // ── Writable data dir ──────────────────────────────────────────
            let data_dir = get_data_dir(app);
            match std::fs::create_dir_all(&data_dir) {
                Ok(_)  => log(&format!("data_dir created OK: {:?}", data_dir)),
                Err(e) => log(&format!("data_dir create FAILED: {:?} -> {}", data_dir, e)),
            }

            // Copy log to the data dir as well for easy discovery
            if let Ok(content) = std::fs::read_to_string(get_log_path()) {
                let _ = std::fs::write(data_dir.join("startup.log"), &content);
            }

            // ── Find Python ────────────────────────────────────────────────
            let python_exe = match find_python() {
                Some(p) => p,
                None => {
                    log("ERROR: Python not found — Flask cannot start");
                    return Ok(());
                }
            };

            // ── Spawn Flask ────────────────────────────────────────────────
            log(&format!("spawning: {} {:?}", python_exe, server_path));

            match Command::new(&python_exe)
                .arg(&server_path)
                .current_dir(&script_dir)
                .env("HATSUN_DATA_DIR", &data_dir)
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
            {
                Ok(child) => {
                    log(&format!("Flask started PID={}", child.id()));
                    if let Ok(mut guard) = FLASK_PROCESS.lock() {
                        *guard = Some(child);
                    }
                }
                Err(e) => {
                    log(&format!("ERROR spawning Flask: {}", e));
                }
            }

            // Copy final log state to data_dir
            if let Ok(content) = std::fs::read_to_string(get_log_path()) {
                let _ = std::fs::write(data_dir.join("startup.log"), &content);
            }

            Ok(())
        })
        .on_window_event(|app, event| {
            if let tauri::WindowEvent::Destroyed = event {
                if let Ok(mut guard) = FLASK_PROCESS.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                    }
                }
                app.app_handle().exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
