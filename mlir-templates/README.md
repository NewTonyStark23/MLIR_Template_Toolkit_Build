# MLIR Template Toolkit

**Safe MLIR code generation that solves Rust's `format!` macro curly brace conflict.**

## The Problem

Rust's `format!` macro treats `{}` as placeholders. MLIR code is **full** of `{}` — for modules, functions, affine maps, linalg regions, tensor types:

```mlir
module @test {                    // ← Rust sees this as a format placeholder
  func.func @main() {             // ← And this
    %0 = linalg.generic {         // ← And this
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>   // ← And these braces
      ]
    }
  }
}
```

Using `format!` on MLIR code causes compile errors or corrupted output because Rust tries to replace **every** `{}` with variables.

## The Solution

Use `{{VAR}}` (double braces) as placeholders. The engine **ONLY** replaces `{{VAR}}` patterns — every single `{` and `}` in the MLIR stays exactly as-is.

```
Template:  module @test { func @main() -> tensor<{{M}}x{{N}}xf16> { ... } }
                        ↑ MLIR brace           ↑ placeholder       ↑ MLIR brace

Rendered:  module @test { func @main() -> tensor<128x256xf16> { ... } }
                        ↑ preserved             ↑ replaced          ↑ preserved
```

## Quick Start

```rust
use mlir_templates::MlirTemplate;
use std::collections::HashMap;

// Load template (embedded at compile time)
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

// Render — ONLY {{VAR}} is replaced, all MLIR {} preserved
let mlir = template.render(&params)?;
```

## How It Works

1. **Scan** — Regex `\{\{(\w+)\}\}` finds all `{{VAR}}` placeholders
2. **Replace** — `str::replace()` substitutes each placeholder with its value
3. **Validate** — Build-time checks ensure balanced braces and valid syntax

**Zero `format!` usage on MLIR content.** The only `format!` call builds the placeholder string `"{{KEY}}"` itself — never applied to the template.

## Crate Structure

```
mlir-templates/
├── Cargo.toml
├── build.rs              # Compile-time template validation
├── src/
│   ├── lib.rs            # Public API exports
│   ├── render.rs         # MlirTemplate engine (core)
│   ├── validate.rs       # Template validation
│   └── error.rs          # Error types
├── templates/
│   ├── matmul_bias_relu.mlir
│   ├── multi_head_attention.mlir
│   ├── layernorm_linear.mlir
│   └── conv_bn_relu.mlir
└── tests/
    └── integration_tests.rs
```

## Included Templates

| Template | Description | Parameters |
|----------|-------------|------------|
| `matmul_bias_relu` | Fused matmul + bias + ReLU | M, N, K, DTYPE |
| `multi_head_attention` | Multi-head self-attention | B, H, S, D, DTYPE |
| `layernorm_linear` | LayerNorm + linear projection | B, S, D, DTYPE, EPS |
| `conv_bn_relu` | Conv2D + BatchNorm + ReLU | B, IC, OC, H, W, KH, KW, DTYPE |

## Validation

Templates are validated at **compile time** via `build.rs`:
- Balanced MLIR braces (single `{` `}`)
- No nested placeholders (`{{{{VAR}}}}`)
- Valid placeholder names (alphanumeric, no leading digits)

## License

MIT OR Apache-2.0
