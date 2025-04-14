use tracing::{info, warn, error, debug, Level};
use tracing_subscriber::FmtSubscriber;
use std::io::{self, Write};

/// Initialize the logger
pub fn init() {
    // Set up the tracing subscriber
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();

    // Set the global default subscriber
    match tracing::subscriber::set_global_default(subscriber) {
        Ok(_) => {},
        Err(e) => {
            eprintln!("Failed to set up logging: {}", e);
        }
    }

    info!("Logger initialized");
}

/// Log an info message
pub fn info(message: &str) {
    info!("{}", message);
}

/// Log a warning message
pub fn warn(message: &str) {
    warn!("{}", message);
}

/// Log an error message
pub fn error(message: &str) {
    error!("{}", message);
}

/// Log a debug message
pub fn debug(message: &str) {
    debug!("{}", message);
}

/// Print a message to stdout immediately
pub fn print(message: &str) {
    println!("{}", message);
}

/// Print a message to stderr immediately
pub fn eprint(message: &str) {
    eprintln!("{}", message);
}