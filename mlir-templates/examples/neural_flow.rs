//! Example: Using MlirTemplate in a neural network codegen context
//!
//! This shows how to replace `format!`-based MLIR generation
//! with the safe template engine.

use mlir_templates::MlirTemplate;
use std::collections::HashMap;

/// Represents a fused operation in a neural network graph
enum FusedOp {
    MatMulBiasRelu { m: usize, n: usize, k: usize },
    LayerNormLinear { batch: usize, seq: usize, dim: usize },
    ConvBnRelu {
        batch: usize,
        in_channels: usize,
        out_channels: usize,
        height: usize,
        width: usize,
    },
}

/// BEFORE (broken): Using format! on MLIR code
///
/// ```rust,ignore
/// // This DOESN'T WORK — Rust's format! tries to interpret
/// // every {} in the MLIR as a placeholder
/// fn generate_mlir_broken(op: &FusedOp) -> String {
///     match op {
///         FusedOp::MatMulBiasRelu { m, n, k } => {
///             format!(r#"
///                 module @matmul {{
///                   func.func @main() -> tensor<{}x{}xf16> {{
///                     // ... lots of MLIR with {} braces ...
///                   }}
///                 }}
///             "#, m, n)
///             // ERROR: Rust sees ALL {} as format placeholders!
///         }
///     }
/// }
/// ```
///
/// AFTER (fixed): Using the template engine
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

        FusedOp::LayerNormLinear { batch, seq, dim } => {
            let template = MlirTemplate::from_embedded(
                "layernorm_linear",
                include_str!("../templates/layernorm_linear.mlir"),
            )?;

            let mut params = HashMap::new();
            params.insert("B".to_string(), batch.to_string());
            params.insert("S".to_string(), seq.to_string());
            params.insert("D".to_string(), dim.to_string());
            params.insert("DTYPE".to_string(), "f16".to_string());
            params.insert("EPS".to_string(), "1e-5".to_string());

            template.render(&params)
        }

        FusedOp::ConvBnRelu {
            batch,
            in_channels,
            out_channels,
            height,
            width,
        } => {
            let template = MlirTemplate::from_embedded(
                "conv_bn_relu",
                include_str!("../templates/conv_bn_relu.mlir"),
            )?;

            let mut params = HashMap::new();
            params.insert("B".to_string(), batch.to_string());
            params.insert("IC".to_string(), in_channels.to_string());
            params.insert("OC".to_string(), out_channels.to_string());
            params.insert("H".to_string(), height.to_string());
            params.insert("W".to_string(), width.to_string());
            params.insert("KH".to_string(), "3".to_string());
            params.insert("KW".to_string(), "3".to_string());
            params.insert("DTYPE".to_string(), "f16".to_string());

            template.render(&params)
        }
    }
}

fn main() {
    // Generate MLIR for a MatMul+Bias+ReLU operation
    let op = FusedOp::MatMulBiasRelu {
        m: 128,
        n: 256,
        k: 512,
    };

    match generate_mlir(&op) {
        Ok(mlir) => {
            println!("Generated MLIR ({} bytes):", mlir.len());
            println!("{}", mlir);
        }
        Err(e) => {
            eprintln!("Error generating MLIR: {}", e);
        }
    }
}
