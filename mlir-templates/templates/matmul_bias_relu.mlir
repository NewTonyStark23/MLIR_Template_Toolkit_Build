// MatMul + Bias + ReLU fused operation
// Parameters: {{M}} (batch), {{N}} (output features), {{K}} (input features), {{DTYPE}} (data type)
//
// This template demonstrates safe {{VAR}} interpolation.
// All structural braces {} are MLIR syntax and must be preserved.

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

    // Step 1: Matrix multiplication C = A * B
    %matmul = linalg.matmul
      ins(%input, %weight : tensor<{{M}}x{{K}}x{{DTYPE}}>, tensor<{{K}}x{{N}}x{{DTYPE}}>)
      outs(%filled : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    // Step 2: Bias addition (broadcast bias over batch dimension)
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

    // Step 3: ReLU activation: max(0, x)
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
}
