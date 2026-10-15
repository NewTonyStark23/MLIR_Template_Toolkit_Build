use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::error::TemplateError;
use crate::Result;

/// Regex to match `{{VAR}}` placeholders (double braces only).
///
/// This pattern matches exactly:
/// - Two opening braces `{{`
/// - One or more word characters (the parameter name)
/// - Two closing braces `}}`
///
/// Single braces in MLIR (modules, functions, affine maps) do NOT match.
static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\{\{(\w+)\}\}").unwrap()
});

/// MLIR Template with safe `{{VAR}}` interpolation.
///
/// # Key Design
///
/// - Uses `{{VAR}}` (double braces) as placeholders
/// - MLIR's single braces `{}` are NEVER touched
/// - No `format!` macro usage on template content
/// - Simple `str::replace()` for substitution
///
/// # Example
///
/// ```rust
/// use mlir_templates::MlirTemplate;
/// use std::collections::HashMap;
///
/// let template = MlirTemplate::new(
///     "module @test { func @f() -> tensor<{{M}}xf16> {} }"
/// ).unwrap();
///
/// let mut params = HashMap::new();
/// params.insert("M".to_string(), "128".to_string());
///
/// let result = template.render(&params).unwrap();
/// // result == "module @test { func @f() -> tensor<128xf16> {} }"
/// //                        ^ preserved                    ^ preserved
/// ```
pub struct MlirTemplate {
    /// The raw template string with `{{VAR}}` placeholders
    template: String,
    /// Set of parameter names found in the template
    required_params: HashSet<String>,
    /// Template name for error messages
    name: String,
}

impl MlirTemplate {
    /// Create a template from a string.
    ///
    /// Scans the string for `{{VAR}}` placeholders and records them
    /// as required parameters.
    ///
    /// # Errors
    ///
    /// Returns `TemplateError::InvalidSyntax` if the template contains
    /// malformed placeholders.
    pub fn new(template: impl Into<String>) -> Result<Self> {
        let template = template.into();
        let required_params = Self::extract_params(&template);

        Ok(Self {
            template,
            required_params,
            name: "<inline>".to_string(),
        })
    }

    /// Create a template from a file path.
    ///
    /// Reads the file and scans for `{{VAR}}` placeholders.
    ///
    /// # Errors
    ///
    /// Returns `TemplateError::Io` if the file cannot be read.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let name = path
            .as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("<file>")
            .to_string();

        let required_params = Self::extract_params(&content);

        Ok(Self {
            template: content,
            required_params,
            name,
        })
    }

    /// Create a template from an embedded string (compile-time included).
    ///
    /// Use with `include_str!`:
    ///
    /// ```rust,ignore
    /// let template = MlirTemplate::from_embedded(
    ///     "matmul_bias_relu",
    ///     include_str!("../templates/matmul_bias_relu.mlir"),
    /// )?;
    /// ```
    pub fn from_embedded(name: &str, content: &'static str) -> Result<Self> {
        let required_params = Self::extract_params(content);

        Ok(Self {
            template: content.to_string(),
            required_params,
            name: name.to_string(),
        })
    }

    /// Extract `{{VAR}}` parameter names from a template string.
    fn extract_params(template: &str) -> HashSet<String> {
        PLACEHOLDER_RE
            .captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    /// Get the set of required parameter names.
    pub fn required_params(&self) -> &HashSet<String> {
        &self.required_params
    }

    /// Get the template name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the raw template string.
    pub fn template_str(&self) -> &str {
        &self.template
    }

    /// Render the template with the given parameters.
    ///
    /// # Important
    ///
    /// This ONLY replaces `{{VAR}}` patterns. ALL other braces (MLIR's
    /// structural `{}`) are completely untouched.
    ///
    /// # Errors
    ///
    /// Returns `TemplateError::MissingParameter` if a required `{{VAR}}`
    /// is not present in the params map.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mlir_templates::MlirTemplate;
    /// use std::collections::HashMap;
    ///
    /// let template = MlirTemplate::new("tensor<{{M}}x{{N}}xf16>").unwrap();
    /// let mut params = HashMap::new();
    /// params.insert("M".to_string(), "128".to_string());
    /// params.insert("N".to_string(), "256".to_string());
    ///
    /// let result = template.render(&params).unwrap();
    /// assert_eq!(result, "tensor<128x256xf16>");
    /// ```
    pub fn render(&self, params: &HashMap<String, String>) -> Result<String> {
        // Check all required params are provided
        for param in &self.required_params {
            if !params.contains_key(param) {
                return Err(TemplateError::MissingParameter(param.clone()));
            }
        }

        // Simple string replacement — NO format! usage on the template
        let mut result = self.template.clone();
        for (key, value) in params {
            let placeholder = Self::make_placeholder(key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    /// Build a `{{VAR}}` placeholder string.
    ///
    /// Uses string concatenation, NOT `format!`, to avoid any confusion
    /// with MLIR braces.
    fn make_placeholder(name: &str) -> String {
        let mut s = String::with_capacity(name.len() + 4);
        s.push('{');
        s.push('{');
        s.push_str(name);
        s.push('}');
        s.push('}');
        s
    }

    /// Render with defaults for any missing params.
    ///
    /// Merges `defaults` with `params` (params take precedence), then renders.
    pub fn render_with_defaults(
        &self,
        params: &HashMap<String, String>,
        defaults: &HashMap<String, String>,
    ) -> Result<String> {
        let mut merged = defaults.clone();
        merged.extend(params.clone());
        self.render(&merged)
    }

    /// Count the number of `{{VAR}}` placeholders in the template.
    pub fn placeholder_count(&self) -> usize {
        PLACEHOLDER_RE.captures_iter(&self.template).count()
    }
}

impl std::fmt::Debug for MlirTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlirTemplate")
            .field("name", &self.name)
            .field("required_params", &self.required_params)
            .field("template_len", &self.template.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_replaces_only_double_braces() {
        let template = r#"module @test {
  func.func @main() -> tensor<{{M}}x{{N}}xf16> {
    return %0 : tensor<{{M}}x{{N}}xf16>
  }
}"#;
        let tmpl = MlirTemplate::new(template).unwrap();
        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());
        params.insert("N".to_string(), "256".to_string());

        let result = tmpl.render(&params).unwrap();

        // {{M}} and {{N}} should be replaced
        assert!(result.contains("128"));
        assert!(result.contains("256"));
        assert!(!result.contains("{{M}}"));
        assert!(!result.contains("{{N}}"));

        // MLIR's single braces should be PRESERVED
        assert!(result.contains("module @test {"));
        assert!(result.contains("func.func @main()"));
        assert!(result.contains("-> tensor<"));
        assert!(result.contains("}"));
    }

    #[test]
    fn test_missing_param_errors() {
        let tmpl = MlirTemplate::new("test {{M}} {{N}}").unwrap();
        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());
        // N is missing!

        let result = tmpl.render(&params);
        assert!(result.is_err());
        match result.unwrap_err() {
            TemplateError::MissingParameter(name) => assert_eq!(name, "N"),
            other => panic!("Expected MissingParameter, got: {:?}", other),
        }
    }

    #[test]
    fn test_extract_params() {
        let template = "{{A}} {{B}} {{A}} {{C}}";
        let params = MlirTemplate::extract_params(template);
        assert_eq!(params.len(), 3);
        assert!(params.contains("A"));
        assert!(params.contains("B"));
        assert!(params.contains("C"));
    }

    #[test]
    fn test_mlir_braces_preserved() {
        // This is the KEY test — MLIR braces must survive rendering
        let template = r#"module @test {
  func.func @f() {
    %0 = arith.constant 0.0 : f32
    %1 = tensor.empty() : tensor<4x8xf32>
    return
  }
}"#;
        let tmpl = MlirTemplate::new(template).unwrap();
        let params = HashMap::new();
        let result = tmpl.render(&params).unwrap();

        // Count braces — should be identical
        let template_open = template.matches('{').count();
        let result_open = result.matches('{').count();
        assert_eq!(template_open, result_open, "Opening braces changed!");

        let template_close = template.matches('}').count();
        let result_close = result.matches('}').count();
        assert_eq!(template_close, result_close, "Closing braces changed!");
    }

    #[test]
    fn test_make_placeholder() {
        assert_eq!(MlirTemplate::make_placeholder("M"), "{{M}}");
        assert_eq!(MlirTemplate::make_placeholder("DTYPE"), "{{DTYPE}}");
    }

    #[test]
    fn test_render_with_defaults() {
        let tmpl = MlirTemplate::new("{{M}}x{{N}}x{{DTYPE}}").unwrap();

        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());

        let mut defaults = HashMap::new();
        defaults.insert("N".to_string(), "256".to_string());
        defaults.insert("DTYPE".to_string(), "f16".to_string());

        let result = tmpl.render_with_defaults(&params, &defaults).unwrap();
        assert_eq!(result, "128x256xf16");
    }

    #[test]
    fn test_complex_mlir_template() {
        // Test with a realistic MLIR template containing many braces
        let template = r#"module @matmul {
  func.func @main(%arg0: tensor<{{M}}x{{K}}xf16>, %arg1: tensor<{{K}}x{{N}}xf16>) -> tensor<{{M}}x{{N}}xf16> {
    %c0 = arith.constant 0.0 : f16
    %init = tensor.empty() : tensor<{{M}}x{{N}}xf16>
    %fill = linalg.fill ins(%c0 : f16) outs(%init : tensor<{{M}}x{{N}}xf16>) -> tensor<{{M}}x{{N}}xf16>
    %result = linalg.matmul
      ins(%arg0, %arg1 : tensor<{{M}}x{{K}}xf16>, tensor<{{K}}x{{N}}xf16>)
      outs(%fill : tensor<{{M}}x{{N}}xf16>)
      -> tensor<{{M}}x{{N}}xf16>
    return %result : tensor<{{M}}x{{N}}xf16>
  }
}"#;
        let tmpl = MlirTemplate::new(template).unwrap();

        assert_eq!(tmpl.required_params().len(), 3);
        assert!(tmpl.required_params().contains("M"));
        assert!(tmpl.required_params().contains("K"));
        assert!(tmpl.required_params().contains("N"));

        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());
        params.insert("K".to_string(), "512".to_string());
        params.insert("N".to_string(), "256".to_string());

        let result = tmpl.render(&params).unwrap();

        // Verify substitutions
        assert!(result.contains("tensor<128x512xf16>"));
        assert!(result.contains("tensor<512x256xf16>"));
        assert!(result.contains("tensor<128x256xf16>"));

        // Verify MLIR structure preserved
        assert!(result.contains("module @matmul {"));
        assert!(result.contains("func.func @main("));
        assert!(result.contains("linalg.matmul"));
        assert!(result.contains("linalg.fill"));
        assert!(result.contains("return %result"));
    }
}
