// Multi-Head Attention
// Parameters: {{B}} (batch), {{H}} (heads), {{S}} (sequence length), {{D}} (head dim), {{DTYPE}}
//
// Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V

module @multi_head_attention {
  func.func @attention(
    %Q: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %K: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %V: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}

    // Step 1: Compute attention scores: scores = Q * K^T
    %scores_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    %scores_filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%scores_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    %scores = linalg.batch_matmul
      ins(%Q, %K : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      outs(%scores_filled : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    // Step 2: Scale by 1/sqrt(d_k)
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

    // Step 3: Softmax + matmul with V (fused via custom call)
    %out_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
    %output = "stablehlo.custom_call"(%scaled, %V) {
      call_target_name = "fused_attention_softmax"
    } : (tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  }
}
