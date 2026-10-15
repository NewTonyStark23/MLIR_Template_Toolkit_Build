//! # MLIR Template Engine
//!
//! Safe string interpolation for MLIR templates using `{{VAR}}` syntax.
//!
//! ## The Problem
//!
//! Rust's `format!` macro treats `{}` as placeholders. MLIR code uses `{}`
//! extensively for modules, functions, affine maps, linalg regions, etc.
//! Using `format!` on MLIR causes compile errors or corrupted output.
//!
//! ## The Solution
//!
//! This crate uses `{{VAR}}` (double braces) as placeholders. The engine
//! ONLY replaces `{{VAR}}` patterns — all MLIR structural braces are untouched.
//!
//! ## Example
//!
//! ```rust
//! use mlir_templates::MlirTemplate;
//! use std::collections::HashMap;
//!
//! let template = MlirTemplate::new("module @test { func @f() -> tensor<{{M}}x{{N}}xf16> {} }").unwrap();
//! let mut params = HashMap::new();
//! params.insert("M".to_string(), "128".to_string());
//! params.insert("N".to_string(), "256".to_string());
//!
//! let result = template.render(&params).unwrap();
//! assert!(result.contains("tensor<128x256xf16>"));
//! assert!(result.contains("module @test {"));  // MLIR braces preserved!
//! ```

pub mod error;
pub mod render;
pub mod validate;

pub use error::TemplateError;
pub use render::MlirTemplate;
pub use validate::{validate_template, ValidationError};

/// Convenience Result type for this crate
pub type Result<T> = std::result::Result<T, TemplateError>;
