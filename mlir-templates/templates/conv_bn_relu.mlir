// Conv2D + BatchNorm + ReLU
// Parameters: {{B}} (batch), {{IC}} (input channels), {{OC}} (output channels),
//             {{H}} (height), {{W}} (width), {{KH}} (kernel h), {{KW}} (kernel w), {{DTYPE}}
//
// Common CNN building block (ResNet-style)

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

    // Step 1: 2D Convolution (NCHW format, FCHW filter)
    %conv = linalg.conv_2d_nchw_fchw {
      strides = dense<[1, 1]> : tensor<2xindex>,
      dilations = dense<[1, 1]> : tensor<2xindex>
    } ins(%input, %filter : tensor<{{B}}x{{IC}}x{{H}}x{{W}}x{{DTYPE}}>, tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    // Step 2: Batch Normalization (inference mode: y = scale * x + bias)
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

    // Step 3: ReLU activation
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
}
