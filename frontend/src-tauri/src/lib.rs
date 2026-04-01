use std::process::{Command, Child};
use std::sync::Mutex;
use tauri::Manager;

static FLASK_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // In dev mode, scripts live at ../../script relative to src-tauri/
            // In production, they are bundled into the resource directory
            let script_dir = if cfg!(debug_assertions) {
                let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
                manifest_dir.join("..").join("..").join("script")
            } else {
                app.path().resource_dir().unwrap().join("script")
            };
            let server_path = script_dir.join("server.py");
            
            println!("Starting Flask from: {:?}", server_path);
            println!("Script dir exists: {}", script_dir.exists());
            println!("server.py exists: {}", server_path.exists());

            // Compute a writable data dir for uploads: AppData\Local\HatsunVRP
            let data_dir = app.path().app_local_data_dir().unwrap_or_else(|_| {
                std::path::PathBuf::from(std::env::var("LOCALAPPDATA").unwrap_or_default())
                    .join("HatsunVRP")
            });
            std::fs::create_dir_all(&data_dir).ok();

            match Command::new("python")
                .arg(&server_path)
                .current_dir(&script_dir)
                .env("HATSUN_DATA_DIR", &data_dir)
                .stdout(std::process::Stdio::inherit())
                .stderr(std::process::Stdio::inherit())
                .spawn()
            {
                Ok(child) => {
                    println!("Flask server started (PID: {})", child.id());
                    if let Ok(mut guard) = FLASK_PROCESS.lock() {
                        *guard = Some(child);
                    }
                }
                Err(e) => eprintln!("Failed to start Flask server: {}", e)
            }
            Ok(())
        })
        .on_window_event(|app, event| {
            // Stop Flask when the window is closed
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
