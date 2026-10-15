// LayerNorm + Linear projection
// Parameters: {{B}} (batch), {{S}} (sequence), {{D}} (hidden dim), {{DTYPE}}, {{EPS}} (epsilon)
//
// Pre-norm transformer block: LayerNorm(x) @ W

module @layernorm_linear {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %gamma: tensor<{{D}}x{{DTYPE}}>,
    %beta: tensor<{{D}}x{{DTYPE}}>,
    %weight: tensor<{{D}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    // Step 1: Layer Normalization
    %normalized = "stablehlo.custom_call"(%input, %gamma, %beta) {
      call_target_name = "layer_norm",
      epsilon = {{EPS}} : {{DTYPE}}
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Step 2: Linear projection
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
}
