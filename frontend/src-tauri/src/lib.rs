use std::process::{Command, Child};
use std::sync::Mutex;
use tauri::Manager;

static FLASK_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            // Find the resource directory where Tauri placed our scripts
            let script_dir = app.path().resource_dir().unwrap().join("script");
            let server_path = script_dir.join("server.py");
            
            println!("Starting Flask from: {:?}", server_path);

            match Command::new("python")
                .arg(&server_path)
                .current_dir(&script_dir)
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
