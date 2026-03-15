// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::{Command, Child};
use std::sync::Mutex;

static FLASK_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

fn start_flask_server() {
    let script_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .map(|p| {
            // In dev: exe is in target/debug, script is in ../../script
            // In prod: exe is alongside the sidecar
            let dev_path = p.join("../../../script");
            let prod_path = p.join("../script");
            if dev_path.join("server.py").exists() {
                dev_path
            } else if prod_path.join("server.py").exists() {
                prod_path
            } else {
                // Fallback: go up from frontend/src-tauri/target/debug
                p.join("../../../../script")
            }
        });

    if let Some(script_dir) = script_dir {
        let server_path = script_dir.join("server.py");
        if server_path.exists() {
            println!("Starting Flask server from: {:?}", server_path);
            match Command::new("python")
                .arg(server_path)
                .current_dir(&script_dir)
                .spawn()
            {
                Ok(child) => {
                    println!("Flask server started (PID: {})", child.id());
                    if let Ok(mut guard) = FLASK_PROCESS.lock() {
                        *guard = Some(child);
                    }
                }
                Err(e) => {
                    eprintln!("Failed to start Flask server: {}", e);
                }
            }
        } else {
            eprintln!("server.py not found at: {:?}", server_path);
        }
    }
}

fn stop_flask_server() {
    if let Ok(mut guard) = FLASK_PROCESS.lock() {
        if let Some(ref mut child) = *guard {
            println!("Stopping Flask server (PID: {})", child.id());
            let _ = child.kill();
            let _ = child.wait();
        }
        *guard = None;
    }
}

fn main() {
    start_flask_server();

    app_lib::run();

    // Cleanup when Tauri closes
    stop_flask_server();
}
