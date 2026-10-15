use std::fs;
use std::path::Path;

fn main() {
    let templates_dir = Path::new("templates");

    if !templates_dir.exists() {
        println!("cargo:warning=No templates/ directory found, skipping validation");
        return;
    }

    let mut has_errors = false;

    for entry in fs::read_dir(templates_dir).expect("Failed to read templates directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "mlir") {
            let content = fs::read_to_string(&path).expect("Failed to read template file");
            let name = path
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();

            match validate_template(&content) {
                Ok(()) => {
                    println!("cargo:warning=  ✓ {}.mlir", name);
                }
                Err(errors) => {
                    has_errors = true;
                    eprintln!();
                    eprintln!("  ✗ {}.mlir:", name);
                    for err in &errors {
                        eprintln!("    - {}", err);
                    }
                }
            }
        }
    }

    if has_errors {
        panic!("Template validation failed! Fix the errors above.");
    }

    // Tell cargo to re-run if templates change
    println!("cargo:rerun-if-changed=templates/");
}

/// Inline validation for build.rs.
///
/// We can't depend on the crate itself in build scripts,
/// so we duplicate the core validation logic here.
fn validate_template(template: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    // Check balanced MLIR braces
    let mut depth: i32 = 0;
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next(); // skip second { (placeholder)
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next(); // skip second } (placeholder)
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push("Unbalanced brace: extra '}'".to_string());
                        depth = 0;
                    }
                }
            }
            _ => {}
        }
    }

    if depth > 0 {
        errors.push(format!("Unbalanced braces: {} unclosed '{{'", depth));
    }

    // Check for nested placeholders
    if template.contains("{{{{") {
        errors.push("Nested placeholders detected (quadruple opening braces)".to_string());
    }
    if template.contains("}}}}") {
        errors.push("Nested placeholders detected (quadruple closing braces)".to_string());
    }

    // Check mismatched double braces
    let double_open = template.matches("{{").count();
    let double_close = template.matches("}}").count();
    if double_open != double_close {
        errors.push(format!(
            "Mismatched double braces: {} opening vs {} closing",
            double_open, double_close
        ));
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
