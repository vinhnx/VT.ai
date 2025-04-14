use std::process::Command;

pub fn run_python_code(code: &str) -> Result<String, String> {
    let output = Command::new("python3")
        .arg("-c")
        .arg(code)
        .output()
        .map_err(|e| e.to_string())?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}