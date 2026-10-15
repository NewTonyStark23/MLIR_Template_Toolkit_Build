use thiserror::Error;

/// Errors that can occur during template operations
#[derive(Error, Debug)]
pub enum TemplateError {
    /// Template file or embedded template not found
    #[error("Template not found: {0}")]
    NotFound(String),

    /// A required {{VAR}} placeholder was not provided in the params map
    #[error("Missing required parameter: {{{{{0}}}}}")]
    MissingParameter(String),

    /// Template syntax is invalid (malformed placeholders, etc.)
    #[error("Invalid template syntax: {0}")]
    InvalidSyntax(String),

    /// Template validation failed (unbalanced braces, etc.)
    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    /// I/O error reading template files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
