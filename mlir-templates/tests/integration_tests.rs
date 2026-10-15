//! Integration tests for the MLIR Template Engine
//!
//! These tests verify the complete workflow: loading templates,
//! rendering with parameters, and validating output.

use mlir_templates::MlirTemplate;
use mlir_templates::validate::{validate_template, count_structural_braces};
use std::collections::HashMap;

/// Test that loading and rendering the matmul template works end-to-end
#[test]
fn test_matmul_template_end_to_end() {
    let template_str = include_str!("../templates/matmul_bias_relu.mlir");

    // Validate at "build time"
    assert!(validate_template(template_str).is_ok());

    // Load and render
    let template = MlirTemplate::from_embedded("matmul_bias_relu", template_str).unwrap();

    let mut params = HashMap::new();
    params.insert("M".to_string(), "128".to_string());
    params.insert("N".to_string(), "256".to_string());
    params.insert("K".to_string(), "512".to_string());
    params.insert("DTYPE".to_string(), "f16".to_string());

    let result = template.render(&params).unwrap();

    // Verify substitutions happened
    assert!(result.contains("tensor<128x512xf16>"), "Input tensor not substituted");
    assert!(result.contains("tensor<512x256xf16>"), "Weight tensor not substituted");
    assert!(result.contains("tensor<128x256xf16>"), "Output tensor not substituted");
    assert!(result.contains("tensor<256xf16>"), "Bias tensor not substituted");

    // Verify NO placeholders remain
    assert!(!result.contains("{{M}}"), "Unresolved {{M}}");
    assert!(!result.contains("{{N}}"), "Unresolved {{N}}");
    assert!(!result.contains("{{K}}"), "Unresolved {{K}}");
    assert!(!result.contains("{{DTYPE}}"), "Unresolved {{DTYPE}}");

    // Verify MLIR structure is intact
    assert!(result.contains("module @matmul_bias_relu {"));
    assert!(result.contains("func.func @main("));
    assert!(result.contains("linalg.matmul"));
    assert!(result.contains("linalg.fill"));
    assert!(result.contains("linalg.generic"));
    assert!(result.contains("affine_map<"));
    assert!(result.contains("return %relu"));
}

/// Test that MLIR braces are preserved through rendering
#[test]
fn test_brace_integrity() {
    let template_str = include_str!("../templates/matmul_bias_relu.mlir");

    let (template_open, template_close) = count_structural_braces(template_str);

    let template = MlirTemplate::from_embedded("matmul_bias_relu", template_str).unwrap();

    let mut params = HashMap::new();
    params.insert("M".to_string(), "64".to_string());
    params.insert("N".to_string(), "128".to_string());
    params.insert("K".to_string(), "256".to_string());
    params.insert("DTYPE".to_string(), "bf16".to_string());

    let result = template.render(&params).unwrap();

    let (result_open, result_close) = count_structural_braces(&result);

    assert_eq!(
        template_open, result_open,
        "Opening structural braces changed: {} -> {}",
        template_open, result_open
    );
    assert_eq!(
        template_close, result_close,
        "Closing structural braces changed: {} -> {}",
        template_close, result_close
    );
}

/// Test that missing parameters produce clear errors
#[test]
fn test_missing_parameter_error() {
    let template = MlirTemplate::new("tensor<{{M}}x{{N}}xf16>").unwrap();

    let mut params = HashMap::new();
    params.insert("M".to_string(), "128".to_string());
    // N is missing!

    let result = template.render(&params);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("N"), "Error should mention missing param N: {}", err_msg);
}

/// Test all included templates are valid
#[test]
fn test_all_templates_valid() {
    let templates = [
        ("matmul_bias_relu", include_str!("../templates/matmul_bias_relu.mlir")),
        ("multi_head_attention", include_str!("../templates/multi_head_attention.mlir")),
        ("layernorm_linear", include_str!("../templates/layernorm_linear.mlir")),
        ("conv_bn_relu", include_str!("../templates/conv_bn_relu.mlir")),
    ];

    for (name, content) in &templates {
        let result = validate_template(content);
        assert!(
            result.is_ok(),
            "Template '{}' failed validation: {:?}",
            name,
            result.err()
        );
    }
}

/// Test that rendering with different parameter values produces different output
#[test]
fn test_different_params_different_output() {
    let template = MlirTemplate::new("tensor<{{M}}x{{N}}x{{DTYPE}}>").unwrap();

    let mut params1 = HashMap::new();
    params1.insert("M".to_string(), "128".to_string());
    params1.insert("N".to_string(), "256".to_string());
    params1.insert("DTYPE".to_string(), "f16".to_string());

    let mut params2 = HashMap::new();
    params2.insert("M".to_string(), "64".to_string());
    params2.insert("N".to_string(), "512".to_string());
    params2.insert("DTYPE".to_string(), "bf16".to_string());

    let result1 = template.render(&params1).unwrap();
    let result2 = template.render(&params2).unwrap();

    assert_ne!(result1, result2);
    assert_eq!(result1, "tensor<128x256xf16>");
    assert_eq!(result2, "tensor<64x512xbf16>");
}

/// Test render_with_defaults
#[test]
fn test_render_with_defaults() {
    let template = MlirTemplate::new("tensor<{{M}}x{{N}}x{{DTYPE}}>").unwrap();

    let mut params = HashMap::new();
    params.insert("M".to_string(), "128".to_string());

    let mut defaults = HashMap::new();
    defaults.insert("N".to_string(), "256".to_string());
    defaults.insert("DTYPE".to_string(), "f16".to_string());

    let result = template.render_with_defaults(&params, &defaults).unwrap();
    assert_eq!(result, "tensor<128x256xf16>");

    // Params should override defaults
    params.insert("N".to_string(), "512".to_string());
    let result2 = template.render_with_defaults(&params, &defaults).unwrap();
    assert_eq!(result2, "tensor<128x512xf16>");
}

/// Test that the engine handles templates with no placeholders
#[test]
fn test_no_placeholders() {
    let template = MlirTemplate::new("module { func @f() { return } }").unwrap();
    assert_eq!(template.required_params().len(), 0);

    let params = HashMap::new();
    let result = template.render(&params).unwrap();
    assert_eq!(result, "module { func @f() { return } }");
}

/// Test that the engine handles templates with many braces
#[test]
fn test_many_braces() {
    let template_str = r#"
module @test {
  func.func @a() {
    %0 = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%in : tensor<{{M}}x{{N}}xf16>)
      outs(%out : tensor<{{M}}x{{N}}xf16>) {
    ^bb0(%a: f16, %b: f16):
      linalg.yield %a : f16
    } -> tensor<{{M}}x{{N}}xf16>
    return
  }
  func.func @b() {
    return
  }
}
"#;

    let template = MlirTemplate::new(template_str).unwrap();
    let mut params = HashMap::new();
    params.insert("M".to_string(), "32".to_string());
    params.insert("N".to_string(), "64".to_string());

    let result = template.render(&params).unwrap();

    // Verify all structural braces preserved
    let (open_before, close_before) = count_structural_braces(template_str);
    let (open_after, close_after) = count_structural_braces(&result);
    assert_eq!(open_before, open_after);
    assert_eq!(close_before, close_after);

    // Verify substitutions
    assert!(result.contains("tensor<32x64xf16>"));
    assert!(!result.contains("{{M}}"));
    assert!(!result.contains("{{N}}"));
}
