use std::fs;
use std::path::Path;

pub fn process_file<P: AsRef<Path>>(path: P) -> Result<String, String> {
    fs::read_to_string(path.as_ref()).map_err(|e| e.to_string())
}