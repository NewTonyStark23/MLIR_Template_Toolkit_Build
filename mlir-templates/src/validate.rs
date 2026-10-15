use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\{\{(\w+)\}\}").unwrap()
});

/// A validation error found in a template
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Human-readable error message
    pub message: String,
    /// Line number where the error was found (if applicable)
    pub line: Option<usize>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {}: {}", line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate a template string.
///
/// Checks:
/// 1. All MLIR structural braces `{` `}` are balanced
/// 2. All `{{VAR}}` placeholders have valid names (alphanumeric, no leading digits)
/// 3. No nested placeholders (`{{{{VAR}}}}`)
/// 4. No orphaned double braces
///
/// # Returns
///
/// `Ok(())` if the template is valid, or `Err` with a list of all errors found.
///
/// # Example
///
/// ```rust
/// use mlir_templates::validate::validate_template;
///
/// // Valid template
/// assert!(validate_template("module { tensor<{{M}}xf16> }").is_ok());
///
/// // Unbalanced braces
/// assert!(validate_template("module { tensor<{{M}}xf16>").is_err());
/// ```
pub fn validate_template(template: &str) -> std::result::Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();

    // Check 1: Balanced MLIR braces
    // We walk through the string tracking brace depth.
    // {{ and }} are skipped (they're placeholders, not MLIR structure).
    let mut depth: i32 = 0;
    let mut chars = template.chars().peekable();
    let mut line: usize = 1;

    while let Some(ch) = chars.next() {
        match ch {
            '\n' => line += 1,
            '{' => {
                if chars.peek() == Some(&'{') {
                    // This is {{ — skip both (it's a placeholder)
                    chars.next();
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    // This is }} — skip both (it's a placeholder)
                    chars.next();
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push(ValidationError {
                            message: "Unbalanced brace: extra '}'".to_string(),
                            line: Some(line),
                        });
                        depth = 0; // Reset to continue checking
                    }
                }
            }
            _ => {}
        }
    }

    if depth > 0 {
        errors.push(ValidationError {
            message: format!("Unbalanced braces: {} unclosed '{{'", depth),
            line: None,
        });
    }

    // Check 2: Valid placeholder names
    for cap in PLACEHOLDER_RE.captures_iter(template) {
        let full_match = cap.get(0).unwrap();
        let name = cap.get(1).unwrap().as_str();

        if name.is_empty() {
            errors.push(ValidationError {
                message: "Empty placeholder name: {{}}".to_string(),
                line: None,
            });
        }

        if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            errors.push(ValidationError {
                message: format!(
                    "Placeholder name cannot start with digit: {}",
                    full_match.as_str()
                ),
                line: None,
            });
        }
    }

    // Check 3: No nested placeholders (quadruple braces)
    if template.contains("{{{{") {
        errors.push(ValidationError {
            message: "Nested placeholders detected (quadruple opening braces '{{{{')".to_string(),
            line: None,
        });
    }
    if template.contains("}}}}") {
        errors.push(ValidationError {
            message: "Nested placeholders detected (quadruple closing braces '}}}}')".to_string(),
            line: None,
        });
    }

    // Check 4: No orphaned double braces (e.g., {{ without }})
    let double_open_count = template.matches("{{").count();
    let double_close_count = template.matches("}}").count();
    if double_open_count != double_close_count {
        errors.push(ValidationError {
            message: format!(
                "Mismatched double braces: {} opening '{{{{' vs {} closing '}}}}'",
                double_open_count, double_close_count
            ),
            line: None,
        });
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Validate a template and extract the set of parameter names.
///
/// Combines validation and parameter extraction in one pass.
///
/// # Returns
///
/// `Ok(HashSet<String>)` with all parameter names if valid,
/// or `Err` with validation errors.
pub fn validate_and_extract(
    template: &str,
) -> std::result::Result<HashSet<String>, Vec<ValidationError>> {
    validate_template(template)?;

    let params: HashSet<String> = PLACEHOLDER_RE
        .captures_iter(template)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();

    Ok(params)
}

/// Count structural (MLIR) braces in a template, ignoring `{{` and `}}`.
///
/// Returns `(open_count, close_count)` of single braces only.
pub fn count_structural_braces(template: &str) -> (usize, usize) {
    let mut open = 0usize;
    let mut close = 0usize;
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next(); // skip second {
                } else {
                    open += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next(); // skip second }
                } else {
                    close += 1;
                }
            }
            _ => {}
        }
    }

    (open, close)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_template() {
        let template = "module { func { tensor<{{M}}xf16> } }";
        assert!(validate_template(template).is_ok());
    }

    #[test]
    fn test_unbalanced_open() {
        let template = "module { func { }";
        let result = validate_template(template);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("unclosed")));
    }

    #[test]
    fn test_unbalanced_close() {
        let template = "module { func } }";
        let result = validate_template(template);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.message.contains("extra")));
    }

    #[test]
    fn test_nested_placeholders() {
        let template = "{{{{VAR}}}}";
        let result = validate_template(template);
        assert!(result.is_err());
    }

    #[test]
    fn test_placeholder_name_starts_with_digit() {
        let template = "{{1VAR}}";
        let result = validate_template(template);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_and_extract() {
        let template = "{{A}} {{B}} {{A}} {{C}}";
        let params = validate_and_extract(template).unwrap();
        assert_eq!(params.len(), 3);
        assert!(params.contains("A"));
        assert!(params.contains("B"));
        assert!(params.contains("C"));
    }

    #[test]
    fn test_count_structural_braces() {
        let template = "module { func { {{M}} } }";
        let (open, close) = count_structural_braces(template);
        assert_eq!(open, 2); // module { and func {
        assert_eq!(close, 2); // two }
    }

    #[test]
    fn test_complex_mlir_valid() {
        let template = r#"module @test {
  func.func @main(%arg0: tensor<{{M}}x{{N}}xf16>) -> tensor<{{M}}x{{N}}xf16> {
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%arg0 : tensor<{{M}}x{{N}}xf16>) {
    ^bb0(%in: f16):
      linalg.yield %in : f16
    } -> tensor<{{M}}x{{N}}xf16>
    return %0 : tensor<{{M}}x{{N}}xf16>
  }
}"#;
        assert!(validate_template(template).is_ok());

        let (open, close) = count_structural_braces(template);
        assert_eq!(open, close, "Structural braces should be balanced");
    }
}
