import { useState } from 'react';
import JSZip from 'jszip';
import { saveAs } from 'file-saver';

export default function App() {
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const zip = new JSZip();
      
      // Add all files to zip
      zip.file('Cargo.toml', CARGO_TOML);
      zip.file('README.md', README);
      zip.file('.gitignore', GITIGNORE);
      zip.file('build.rs', BUILD_RS);
      zip.file('src/lib.rs', LIB_RS);
      zip.file('src/render.rs', RENDER_RS);
      zip.file('src/validate.rs', VALIDATE_RS);
      zip.file('src/error.rs', ERROR_RS);
      zip.file('templates/matmul_bias_relu.mlir', MATMUL_TEMPLATE);
      zip.file('templates/multi_head_attention.mlir', ATTENTION_TEMPLATE);
      zip.file('templates/layernorm_linear.mlir', LAYERNORM_TEMPLATE);
      zip.file('templates/conv_bn_relu.mlir', CONV_TEMPLATE);
      zip.file('tests/integration_tests.rs', INTEGRATION_TESTS);
      zip.file('examples/neural_flow.rs', EXAMPLE);
      
      // Generate and download
      const blob = await zip.generateAsync({ type: 'blob' });
      saveAs(blob, 'mlir-templates.zip');
    } catch (error) {
      console.error('Download failed:', error);
      alert('Download failed. Please try again.');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-950 to-black text-gray-100">
      {/* Hero Section */}
      <div className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto px-4 py-12">
          <div className="text-center space-y-6">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-gradient-to-br from-emerald-400 to-cyan-500 shadow-2xl shadow-emerald-500/30">
              <svg className="w-12 h-12 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            </div>
            
            <h1 className="text-5xl font-bold bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-400 bg-clip-text text-transparent">
              MLIR Template Toolkit
            </h1>
            
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Safe <code className="px-2 py-1 bg-gray-800 rounded text-emerald-400">{'{{VAR}}'}</code> interpolation for MLIR code generation.
              <br />
              Solves Rust's <code className="px-2 py-1 bg-gray-800 rounded text-red-400">format!</code> macro curly brace conflicts.
            </p>

            {/* DOWNLOAD BUTTON */}
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold text-lg rounded-xl shadow-2xl shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all transform hover:scale-105 disabled:scale-100"
            >
              {downloading ? (
                <>
                  <svg className="animate-spin w-6 h-6" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Packaging...
                </>
              ) : (
                <>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download Complete Crate (.zip)
                </>
              )}
            </button>

            <p className="text-sm text-gray-500">
              Includes: Cargo.toml, src/, templates/, tests/, examples/, build.rs, README.md
            </p>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-12 space-y-12">
        {/* Problem/Solution */}
        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-red-950/20 border border-red-900/50 rounded-xl p-6">
            <h3 className="text-xl font-bold text-red-300 mb-3 flex items-center gap-2">
              <span className="text-2xl">❌</span> The Problem
            </h3>
            <pre className="text-xs font-mono text-gray-400 bg-gray-900/50 rounded-lg p-4 overflow-x-auto">
{`// Rust's format! treats ALL {} as placeholders
format!(r#"
  module @test {{
    func @main() -> tensor<{}x{}xf16> {{
      // Rust eats these braces!
    }}
  }}
"#, m, n)

// Result: Compile error or corrupted MLIR`}
            </pre>
          </div>

          <div className="bg-emerald-950/20 border border-emerald-900/50 rounded-xl p-6">
            <h3 className="text-xl font-bold text-emerald-300 mb-3 flex items-center gap-2">
              <span className="text-2xl">✅</span> The Solution
            </h3>
            <pre className="text-xs font-mono text-gray-400 bg-gray-900/50 rounded-lg p-4 overflow-x-auto">
{`// Engine ONLY replaces {{VAR}}
let template = MlirTemplate::new(r#"
  module @test {
    func @main() -> tensor<{{M}}x{{N}}xf16> {
      // These braces are safe!
    }
  }
"#)?;
template.render(&params)?;

// Result: Valid MLIR every time`}
            </pre>
          </div>
        </div>

        {/* Features */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6 text-center">What's Included</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {[
              { icon: '⚙️', title: 'Core Engine', desc: 'MlirTemplate struct with render() method' },
              { icon: '✅', title: 'Validation', desc: 'Compile-time template checking via build.rs' },
              { icon: '📦', title: '4 Templates', desc: 'MatMul, Attention, LayerNorm, Conv2D' },
              { icon: '🧪', title: 'Tests', desc: 'Integration tests with brace integrity checks' },
              { icon: '💡', title: 'Examples', desc: 'Usage in neural network codegen' },
              { icon: '📖', title: 'Documentation', desc: 'Complete README with usage guide' },
            ].map((feature, i) => (
              <div key={i} className="flex items-start gap-3">
                <span className="text-3xl">{feature.icon}</span>
                <div>
                  <h3 className="font-semibold text-gray-200">{feature.title}</h3>
                  <p className="text-sm text-gray-400">{feature.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Quick Start */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-4">Quick Start</h2>
          <pre className="text-sm font-mono text-gray-300 bg-gray-800/50 rounded-lg p-6 overflow-x-auto">
{`# 1. Download and extract
unzip mlir-templates.zip
cd mlir-templates

# 2. Build (templates validated at compile time)
cargo build

# 3. Run tests
cargo test

# 4. Use in your project
# Add to your Cargo.toml:
[dependencies]
mlir-templates = { path = "../mlir-templates" }`}
          </pre>
        </div>

        {/* Usage Example */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-4">Usage Example</h2>
          <pre className="text-sm font-mono text-gray-300 bg-gray-800/50 rounded-lg p-6 overflow-x-auto">
{`use mlir_templates::MlirTemplate;
use std::collections::HashMap;

// Load template
let template = MlirTemplate::from_embedded(
    "matmul_bias_relu",
    include_str!("../templates/matmul_bias_relu.mlir"),
)?;

// Set parameters
let mut params = HashMap::new();
params.insert("M".to_string(), "128".to_string());
params.insert("N".to_string(), "256".to_string());
params.insert("K".to_string(), "512".to_string());
params.insert("DTYPE".to_string(), "f16".to_string());

// Render — ONLY {{VAR}} replaced, all MLIR {} preserved
let mlir = template.render(&params)?;`}
          </pre>
        </div>

        {/* File Structure */}
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-4">Repository Structure</h2>
          <pre className="text-sm font-mono text-gray-400 bg-gray-800/50 rounded-lg p-6 overflow-x-auto">
{`mlir-templates/
├── Cargo.toml              # Package manifest
├── README.md               # Documentation
├── build.rs                # Compile-time validation
├── src/
│   ├── lib.rs              # Public API
│   ├── render.rs           # MlirTemplate engine
│   ├── validate.rs         # Validation logic
│   └── error.rs            # Error types
├── templates/
│   ├── matmul_bias_relu.mlir
│   ├── multi_head_attention.mlir
│   ├── layernorm_linear.mlir
│   └── conv_bn_relu.mlir
├── tests/
│   └── integration_tests.rs
└── examples/
    └── neural_flow.rs`}
          </pre>
        </div>

        {/* Bottom CTA */}
        <div className="text-center py-8">
          <button
            onClick={handleDownload}
            disabled={downloading}
            className="inline-flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold text-lg rounded-xl shadow-2xl shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all transform hover:scale-105 disabled:scale-100"
          >
            {downloading ? (
              <>
                <svg className="animate-spin w-6 h-6" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Packaging...
              </>
            ) : (
              <>
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Complete Crate (.zip)
              </>
            )}
          </button>
        </div>
      </div>

      <footer className="border-t border-gray-800 mt-12 py-6 text-center text-sm text-gray-600">
        MLIR Template Toolkit — Solves Rust format! macro conflicts with MLIR curly braces
      </footer>
    </div>
  );
}

// ============================================================================
// FILE CONTENTS (embedded in the app for download)
// ============================================================================

const CARGO_TOML = `[package]
name = "mlir-templates"
version = "0.1.0"
edition = "2021"
description = "Safe MLIR template engine — {{VAR}} interpolation that never touches MLIR's structural braces"
license = "MIT OR Apache-2.0"
repository = "https://github.com/your-org/mlir-templates"
readme = "README.md"
keywords = ["mlir", "template", "compiler", "codegen", "xla"]
categories = ["compilers", "template-engine"]

[dependencies]
thiserror = "1.0"
regex = "1.10"
once_cell = "1.19"

[dev-dependencies]
tempfile = "3.8"

[build-dependencies]
regex = "1.10"

[features]
default = ["std"]
std = []`;

const README = `# MLIR Template Toolkit

**Safe MLIR code generation that solves Rust's \`format!\` macro curly brace conflict.**

## The Problem

Rust's \`format!\` macro treats \`{}\` as placeholders. MLIR code is **full** of \`{}\` — for modules, functions, affine maps, linalg regions, tensor types. Using \`format!\` on MLIR causes compile errors or corrupted output.

## The Solution

Use \`{{VAR}}\` (double braces) as placeholders. The engine **ONLY** replaces \`{{VAR}}\` patterns — every single \`{\` and \`}\` in the MLIR stays exactly as-is.

## Quick Start

\`\`\`rust
use mlir_templates::MlirTemplate;
use std::collections::HashMap;

let template = MlirTemplate::from_embedded(
    "matmul_bias_relu",
    include_str!("../templates/matmul_bias_relu.mlir"),
)?;

let mut params = HashMap::new();
params.insert("M".to_string(), "128".to_string());
params.insert("N".to_string(), "256".to_string());
params.insert("K".to_string(), "512".to_string());
params.insert("DTYPE".to_string(), "f16".to_string());

let mlir = template.render(&params)?;
\`\`\`

## How It Works

1. **Scan** — Regex \`\\{\\{(\\w+)\\}\\}\` finds all \`{{VAR}}\` placeholders
2. **Replace** — \`str::replace()\` substitutes each placeholder
3. **Validate** — Build-time checks ensure balanced braces

**Zero \`format!\` usage on MLIR content.**

## License

MIT OR Apache-2.0`;

const GITIGNORE = `/target
Cargo.lock
*.swp
*.swo
*~
.idea/
.vscode/
.DS_Store`;

const BUILD_RS = `use std::fs;
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
            let name = path.file_stem().unwrap().to_str().unwrap().to_string();

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

    println!("cargo:rerun-if-changed=templates/");
}

fn validate_template(template: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut depth: i32 = 0;
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
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

    if template.contains("{{{{") {
        errors.push("Nested placeholders detected".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}`;

const LIB_RS = `//! # MLIR Template Engine
//!
//! Safe string interpolation for MLIR templates using \`{{VAR}}\` syntax.

pub mod error;
pub mod render;
pub mod validate;

pub use error::TemplateError;
pub use render::MlirTemplate;
pub use validate::{validate_template, ValidationError};

pub type Result<T> = std::result::Result<T, TemplateError>;`;

const ERROR_RS = `use thiserror::Error;

#[derive(Error, Debug)]
pub enum TemplateError {
    #[error("Template not found: {0}")]
    NotFound(String),

    #[error("Missing required parameter: {{{{{0}}}}}")]
    MissingParameter(String),

    #[error("Invalid template syntax: {0}")]
    InvalidSyntax(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}`;

const RENDER_RS = `use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::error::TemplateError;
use crate::Result;

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

pub struct MlirTemplate {
    template: String,
    required_params: HashSet<String>,
    name: String,
}

impl MlirTemplate {
    pub fn new(template: impl Into<String>) -> Result<Self> {
        let template = template.into();
        let required_params = Self::extract_params(&template);

        Ok(Self {
            template,
            required_params,
            name: "<inline>".to_string(),
        })
    }

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

    pub fn from_embedded(name: &str, content: &'static str) -> Result<Self> {
        let required_params = Self::extract_params(content);

        Ok(Self {
            template: content.to_string(),
            required_params,
            name: name.to_string(),
        })
    }

    fn extract_params(template: &str) -> HashSet<String> {
        PLACEHOLDER_RE
            .captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }

    pub fn required_params(&self) -> &HashSet<String> {
        &self.required_params
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn render(&self, params: &HashMap<String, String>) -> Result<String> {
        for param in &self.required_params {
            if !params.contains_key(param) {
                return Err(TemplateError::MissingParameter(param.clone()));
            }
        }

        let mut result = self.template.clone();
        for (key, value) in params {
            let placeholder = Self::make_placeholder(key);
            result = result.replace(&placeholder, value);
        }

        Ok(result)
    }

    fn make_placeholder(name: &str) -> String {
        let mut s = String::with_capacity(name.len() + 4);
        s.push('{');
        s.push('{');
        s.push_str(name);
        s.push('}');
        s.push('}');
        s
    }

    pub fn render_with_defaults(
        &self,
        params: &HashMap<String, String>,
        defaults: &HashMap<String, String>,
    ) -> Result<String> {
        let mut merged = defaults.clone();
        merged.extend(params.clone());
        self.render(&merged)
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

        assert!(result.contains("128"));
        assert!(result.contains("256"));
        assert!(!result.contains("{{M}}"));
        assert!(!result.contains("{{N}}"));
        assert!(result.contains("module @test {"));
        assert!(result.contains("func.func @main()"));
    }

    #[test]
    fn test_missing_param_errors() {
        let tmpl = MlirTemplate::new("test {{M}} {{N}}").unwrap();
        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());

        let result = tmpl.render(&params);
        assert!(result.is_err());
    }
}`;

const VALIDATE_RS = `use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub message: String,
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

pub fn validate_template(template: &str) -> std::result::Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    let mut depth: i32 = 0;
    let mut chars = template.chars().peekable();
    let mut line: usize = 1;

    while let Some(ch) = chars.next() {
        match ch {
            '\\n' => line += 1,
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push(ValidationError {
                            message: "Unbalanced brace: extra '}'".to_string(),
                            line: Some(line),
                        });
                        depth = 0;
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

    for cap in PLACEHOLDER_RE.captures_iter(template) {
        let name = cap.get(1).unwrap().as_str();
        if name.is_empty() {
            errors.push(ValidationError {
                message: "Empty placeholder name: {{}}".to_string(),
                line: None,
            });
        }
        if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            errors.push(ValidationError {
                message: format!("Placeholder name cannot start with digit: {{{{{}}}}}", name),
                line: None,
            });
        }
    }

    if template.contains("{{{{") || template.contains("}}}}") {
        errors.push(ValidationError {
            message: "Nested placeholders detected".to_string(),
            line: None,
        });
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn validate_and_extract(
    template: &str,
) -> std::result::Result<HashSet<String>, Vec<ValidationError>> {
    validate_template(template)?;

    let params: HashSet<String> = PLACEHOLDER_RE
        .captures_iter(template)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();

    Ok(params)
}`;

const MATMUL_TEMPLATE = `// MatMul + Bias + ReLU fused operation
// Parameters: {{M}} (batch), {{N}} (output features), {{K}} (input features), {{DTYPE}}

module @matmul_bias_relu {
  func.func @main(
    %input: tensor<{{M}}x{{K}}x{{DTYPE}}>,
    %weight: tensor<{{K}}x{{N}}x{{DTYPE}}>,
    %bias: tensor<{{N}}x{{DTYPE}}>
  ) -> tensor<{{M}}x{{N}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %init = tensor.empty() : tensor<{{M}}x{{N}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    %matmul = linalg.matmul
      ins(%input, %weight : tensor<{{M}}x{{K}}x{{DTYPE}}>, tensor<{{K}}x{{N}}x{{DTYPE}}>)
      outs(%filled : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    %bias_out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul, %bias : tensor<{{M}}x{{N}}x{{DTYPE}}>, tensor<{{N}}x{{DTYPE}}>)
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) {
    ^bb0(%a: {{DTYPE}}, %b: {{DTYPE}}, %out: {{DTYPE}}):
      %sum = arith.addf %a, %b : {{DTYPE}}
      linalg.yield %sum : {{DTYPE}}
    } -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias_out : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %max = arith.maximumf %in, %c0 : {{DTYPE}}
      linalg.yield %max : {{DTYPE}}
    } -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    return %relu : tensor<{{M}}x{{N}}x{{DTYPE}}>
  }
}`;

const ATTENTION_TEMPLATE = `// Multi-Head Attention
// Parameters: {{B}} (batch), {{H}} (heads), {{S}} (sequence), {{D}} (head dim), {{DTYPE}}

module @multi_head_attention {
  func.func @attention(
    %Q: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %K: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %V: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}

    %scores_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    %scores = linalg.batch_matmul
      ins(%Q, %K : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      outs(%scores_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    %scale = arith.constant 0.125 : {{DTYPE}}
    %scaled = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, i, j) -> (b, h, i, j)>,
        affine_map<(b, h, i, j) -> (b, h, i, j)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scores : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      outs(%scores_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %mul = arith.mulf %in, %scale : {{DTYPE}}
      linalg.yield %mul : {{DTYPE}}
    } -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    %out_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
    %output = "stablehlo.custom_call"(%scaled, %V) {
      call_target_name = "fused_attention_softmax"
    } : (tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  }
}`;

const LAYERNORM_TEMPLATE = `// LayerNorm + Linear projection
// Parameters: {{B}} (batch), {{S}} (sequence), {{D}} (hidden dim), {{DTYPE}}, {{EPS}}

module @layernorm_linear {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %gamma: tensor<{{D}}x{{DTYPE}}>,
    %beta: tensor<{{D}}x{{DTYPE}}>,
    %weight: tensor<{{D}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    %normalized = "stablehlo.custom_call"(%input, %gamma, %beta) {
      call_target_name = "layer_norm",
      epsilon = {{EPS}} : {{DTYPE}}
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    %c0 = arith.constant 0.0 : {{DTYPE}}
    %out = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%out : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    %result = linalg.batch_matmul
      ins(%normalized, %weight : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %result : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
}`;

const CONV_TEMPLATE = `// Conv2D + BatchNorm + ReLU
// Parameters: {{B}}, {{IC}}, {{OC}}, {{H}}, {{W}}, {{KH}}, {{KW}}, {{DTYPE}}

module @conv_bn_relu {
  func.func @main(
    %input: tensor<{{B}}x{{IC}}x{{H}}x{{W}}x{{DTYPE}}>,
    %filter: tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>,
    %bn_scale: tensor<{{OC}}x{{DTYPE}}>,
    %bn_bias: tensor<{{OC}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %conv_out = tensor.empty() : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%conv_out : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    %conv = linalg.conv_2d_nchw_fchw {
      strides = dense<[1, 1]> : tensor<2xindex>,
      dilations = dense<[1, 1]> : tensor<2xindex>
    } ins(%input, %filter : tensor<{{B}}x{{IC}}x{{H}}x{{W}}x{{DTYPE}}>, tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    %bn = linalg.generic {
      indexing_maps = [
        affine_map<(b, c, h, w) -> (b, c, h, w)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (b, c, h, w)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv, %bn_scale, %bn_bias :
      tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>, tensor<{{OC}}x{{DTYPE}}>, tensor<{{OC}}x{{DTYPE}}>)
      outs(%conv_out : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %s: {{DTYPE}}, %b: {{DTYPE}}, %out: {{DTYPE}}):
      %mul = arith.mulf %in, %s : {{DTYPE}}
      %add = arith.addf %mul, %b : {{DTYPE}}
      linalg.yield %add : {{DTYPE}}
    } -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    %relu = linalg.generic {
      indexing_maps = [
        affine_map<(b, c, h, w) -> (b, c, h, w)>,
        affine_map<(b, c, h, w) -> (b, c, h, w)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bn : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      outs(%conv_out : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %max = arith.maximumf %in, %c0 : {{DTYPE}}
      linalg.yield %max : {{DTYPE}}
    } -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    return %relu : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>
  }
}`;

const INTEGRATION_TESTS = `use mlir_templates::MlirTemplate;
use mlir_templates::validate::validate_template;
use std::collections::HashMap;

#[test]
fn test_matmul_template_end_to_end() {
    let template_str = include_str!("../templates/matmul_bias_relu.mlir");
    assert!(validate_template(template_str).is_ok());

    let template = MlirTemplate::from_embedded("matmul_bias_relu", template_str).unwrap();

    let mut params = HashMap::new();
    params.insert("M".to_string(), "128".to_string());
    params.insert("N".to_string(), "256".to_string());
    params.insert("K".to_string(), "512".to_string());
    params.insert("DTYPE".to_string(), "f16".to_string());

    let result = template.render(&params).unwrap();

    assert!(result.contains("tensor<128x512xf16>"));
    assert!(result.contains("tensor<512x256xf16>"));
    assert!(result.contains("tensor<128x256xf16>"));
    assert!(!result.contains("{{M}}"));
    assert!(!result.contains("{{N}}"));
    assert!(!result.contains("{{K}}"));
    assert!(!result.contains("{{DTYPE}}"));
    assert!(result.contains("module @matmul_bias_relu {"));
    assert!(result.contains("func.func @main("));
    assert!(result.contains("linalg.matmul"));
}`;

const EXAMPLE = `use mlir_templates::MlirTemplate;
use std::collections::HashMap;

enum FusedOp {
    MatMulBiasRelu { m: usize, n: usize, k: usize },
}

fn generate_mlir(op: &FusedOp) -> mlir_templates::Result<String> {
    match op {
        FusedOp::MatMulBiasRelu { m, n, k } => {
            let template = MlirTemplate::from_embedded(
                "matmul_bias_relu",
                include_str!("../templates/matmul_bias_relu.mlir"),
            )?;

            let mut params = HashMap::new();
            params.insert("M".to_string(), m.to_string());
            params.insert("N".to_string(), n.to_string());
            params.insert("K".to_string(), k.to_string());
            params.insert("DTYPE".to_string(), "f16".to_string());

            template.render(&params)
        }
    }
}

fn main() {
    let op = FusedOp::MatMulBiasRelu { m: 128, n: 256, k: 512 };

    match generate_mlir(&op) {
        Ok(mlir) => {
            println!("Generated MLIR ({} bytes):", mlir.len());
            println!("{}", mlir);
        }
        Err(e) => {
            eprintln!("Error: {}", e);
        }
    }
}`;
