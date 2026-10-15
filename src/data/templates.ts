import { TemplateParam } from '../utils/engine';

export interface MlirTemplateDef {
  id: string;
  name: string;
  category: string;
  description: string;
  template: string;
  params: TemplateParam[];
  tags: string[];
}

export const TEMPLATE_CATEGORIES = [
  'Linear Algebra',
  'Attention',
  'Normalization',
  'Convolution',
  'Embedding',
  'Custom',
];

export const TEMPLATES: MlirTemplateDef[] = [
  {
    id: 'matmul-bias-relu',
    name: 'MatMul + Bias + ReLU',
    category: 'Linear Algebra',
    description: 'Matrix multiplication followed by bias addition and ReLU activation. Common in fully-connected layers.',
    tags: ['gemm', 'linear', 'fc', 'activation'],
    params: [
      { name: 'M', value: '', description: 'Batch dimension (rows of output)', defaultValue: '128', type: 'dimension' },
      { name: 'N', value: '', description: 'Output features dimension', defaultValue: '256', type: 'dimension' },
      { name: 'K', value: '', description: 'Input features dimension (inner dim)', defaultValue: '512', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type for tensors', defaultValue: 'f16', type: 'dtype' },
      { name: 'TILE_M', value: '', description: 'Tile size for M dimension', defaultValue: '64', type: 'integer' },
      { name: 'TILE_N', value: '', description: 'Tile size for N dimension', defaultValue: '64', type: 'integer' },
      { name: 'TILE_K', value: '', description: 'Tile size for K dimension', defaultValue: '32', type: 'integer' },
    ],
    template: `module @matmul_bias_relu {
  func.func @main(%input: tensor<{{M}}x{{K}}x{{DTYPE}}>,
                  %weight: tensor<{{K}}x{{N}}x{{DTYPE}}>,
                  %bias: tensor<{{N}}x{{DTYPE}}>) -> tensor<{{M}}x{{N}}x{{DTYPE}}> {
    // Matrix multiplication: C = A * B
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %init = tensor.empty() : tensor<{{M}}x{{N}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) -> tensor<{{M}}x{{N}}x{{DTYPE}}>
    
    %matmul = linalg.matmul
      ins(%input, %weight : tensor<{{M}}x{{K}}x{{DTYPE}}>, tensor<{{K}}x{{N}}x{{DTYPE}}>)
      outs(%filled : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    // Bias addition: C = matmul + bias (broadcast)
    %bias_result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%matmul, %bias : tensor<{{M}}x{{N}}x{{DTYPE}}>, tensor<{{N}}x{{DTYPE}}>)
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) {
    ^bb0(%in1: {{DTYPE}}, %in2: {{DTYPE}}, %out: {{DTYPE}}):
      %add = arith.addf %in1, %in2 : {{DTYPE}}
      linalg.yield %add : {{DTYPE}}
    } -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    // ReLU activation: max(0, x)
    %relu_result = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%bias_result : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %max = arith.maximumf %in, %c0 : {{DTYPE}}
      linalg.yield %max : {{DTYPE}}
    } -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    return %relu_result : tensor<{{M}}x{{N}}x{{DTYPE}}>
  }

  // Tiling configuration: {{TILE_M}}x{{TILE_N}}x{{TILE_K}}
  // Target: GPU/CPU with tile-based execution
}`,
  },
  {
    id: 'multi-head-attention',
    name: 'Multi-Head Attention',
    category: 'Attention',
    description: 'Scaled dot-product multi-head attention mechanism. Core of transformer architectures.',
    tags: ['transformer', 'attention', 'mha', 'self-attention'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '512', type: 'dimension' },
      { name: 'H', value: '', description: 'Number of attention heads', defaultValue: '8', type: 'integer' },
      { name: 'D', value: '', description: 'Head dimension (d_k)', defaultValue: '64', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'f16', type: 'dtype' },
      { name: 'SCALE', value: '', description: 'Attention scale factor (1/sqrt(d_k))', defaultValue: '0.125', type: 'string' },
    ],
    template: `module @multi_head_attention {
  func.func @attention(
    %query: tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>,
    %key: tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>,
    %value: tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}> {
    // Step 1: QK^T - Compute attention scores
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %score_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    %score_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%score_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>) -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    
    %scores = linalg.batch_matmul
      ins(%query, %key : tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>)
      outs(%score_filled : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    // Step 2: Scale scores by 1/sqrt(d_k) = {{SCALE}}
    %scale_val = arith.constant {{SCALE}} : {{DTYPE}}
    %scaled = linalg.generic {
      indexing_maps = [
        affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>,
        affine_map<(b, h, s1, s2) -> (b, h, s1, s2)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%scores : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      outs(%score_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %mul = arith.mulf %in, %scale_val : {{DTYPE}}
      linalg.yield %mul : {{DTYPE}}
    } -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    // Step 3: Softmax over last dimension
    %softmax = func.call @softmax_op(%scaled) 
      : (tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>) -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    // Step 4: Attention output = softmax(scores) * V
    %out_init = tensor.empty() : tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>
    %out_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%out_init : tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>
    
    %output = linalg.batch_matmul
      ins(%softmax, %value : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>, tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>)
      outs(%out_filled : tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{S}}x{{H}}x{{D}}x{{DTYPE}}>
  }

  // Configuration: B={{B}}, S={{S}}, H={{H}}, D={{D}}
  // Total params per layer: 4 * {{H}} * {{D}} * {{D}} (Q, K, V, O projections)
  // FLOPs: 2 * {{B}} * {{H}} * {{S}} * {{S}} * {{D}} (attention) + 2 * {{B}} * {{H}} * {{S}} * {{D}} * {{D}} (output proj)
}`,
  },
  {
    id: 'layernorm-linear',
    name: 'LayerNorm + Linear',
    category: 'Normalization',
    description: 'Layer normalization followed by a linear projection. Used in transformer blocks (pre-norm architecture).',
    tags: ['layernorm', 'linear', 'transformer', 'pre-norm'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '512', type: 'dimension' },
      { name: 'D', value: '', description: 'Hidden dimension', defaultValue: '1024', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'f16', type: 'dtype' },
      { name: 'EPSILON', value: '', description: 'LayerNorm epsilon', defaultValue: '1e-5', type: 'string' },
    ],
    template: `module @layernorm_linear {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %gamma: tensor<{{D}}x{{DTYPE}}>,
    %beta: tensor<{{D}}x{{DTYPE}}>,
    %weight: tensor<{{D}}x{{D}}x{{DTYPE}}>,
    %bias: tensor<{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    // Step 1: Layer Normalization
    // mean = reduce_sum(input) / D
    // var = reduce_sum((input - mean)^2) / D
    // output = gamma * (input - mean) / sqrt(var + epsilon) + beta
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %ceps = arith.constant {{EPSILON}} : {{DTYPE}}
    %norm_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    %normalized = "stablehlo.custom_call"(%input, %gamma, %beta) {
      call_target_name = "layer_norm",
      epsilon = {{EPSILON}} : {{DTYPE}}
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>) 
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Step 2: Linear projection
    %proj_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %proj_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%proj_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    %output = linalg.batch_matmul
      ins(%normalized, %weight : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D}}x{{DTYPE}}>)
      outs(%proj_filled : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Add bias
    %result = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, d) -> (b, s, d)>,
        affine_map<(b, s, d) -> (d)>,
        affine_map<(b, s, d) -> (b, s, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%output, %bias : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>)
      outs(%proj_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) {
    ^bb0(%in1: {{DTYPE}}, %in2: {{DTYPE}}, %out: {{DTYPE}}):
      %add = arith.addf %in1, %in2 : {{DTYPE}}
      linalg.yield %add : {{DTYPE}}
    } -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %result : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
  
  // Config: B={{B}}, S={{S}}, D={{D}}, eps={{EPSILON}}
}`,
  },
  {
    id: 'rmsnorm-linear',
    name: 'RMSNorm + Linear',
    category: 'Normalization',
    description: 'Root Mean Square Layer Normalization followed by linear. Used in LLaMA and modern transformer architectures.',
    tags: ['rmsnorm', 'llama', 'linear', 'normalization'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '512', type: 'dimension' },
      { name: 'D', value: '', description: 'Hidden dimension', defaultValue: '4096', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'bf16', type: 'dtype' },
      { name: 'EPSILON', value: '', description: 'RMSNorm epsilon', defaultValue: '1e-6', type: 'string' },
    ],
    template: `module @rmsnorm_linear {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %weight: tensor<{{D}}x{{DTYPE}}>,
    %proj_weight: tensor<{{D}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    // Step 1: RMS Normalization
    // rms = sqrt(mean(x^2) + epsilon)
    // output = x / rms * weight
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %c1 = arith.constant 1.0 : {{DTYPE}}
    %norm_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    // Compute x^2
    %squared = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, d) -> (b, s, d)>,
        affine_map<(b, s, d) -> (b, s, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%input : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      outs(%norm_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %sq = arith.mulf %in, %in : {{DTYPE}}
      linalg.yield %sq : {{DTYPE}}
    } -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Reduce mean over D dimension, add epsilon, rsqrt
    %rms_norm = "stablehlo.custom_call"(%input, %weight) {
      call_target_name = "rms_norm",
      epsilon = {{EPSILON}} : {{DTYPE}}
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Step 2: Linear projection
    %proj_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %proj_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%proj_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    %output = linalg.batch_matmul
      ins(%rms_norm, %proj_weight : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D}}x{{DTYPE}}>)
      outs(%proj_filled : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
  
  // Config: B={{B}}, S={{S}}, D={{D}}, eps={{EPSILON}}
  // RMSNorm is cheaper than LayerNorm (no mean subtraction)
}`,
  },
  {
    id: 'conv-bn-relu',
    name: 'Conv2D + BatchNorm + ReLU',
    category: 'Convolution',
    description: '2D convolution followed by batch normalization and ReLU activation. Common in CNN architectures like ResNet.',
    tags: ['conv2d', 'batchnorm', 'relu', 'cnn', 'resnet'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '8', type: 'dimension' },
      { name: 'IH', value: '', description: 'Input height', defaultValue: '224', type: 'dimension' },
      { name: 'IW', value: '', description: 'Input width', defaultValue: '224', type: 'dimension' },
      { name: 'IC', value: '', description: 'Input channels', defaultValue: '64', type: 'dimension' },
      { name: 'OC', value: '', description: 'Output channels (filters)', defaultValue: '128', type: 'dimension' },
      { name: 'KH', value: '', description: 'Kernel height', defaultValue: '3', type: 'dimension' },
      { name: 'KW', value: '', description: 'Kernel width', defaultValue: '3', type: 'dimension' },
      { name: 'SH', value: '', description: 'Stride height', defaultValue: '1', type: 'integer' },
      { name: 'SW', value: '', description: 'Stride width', defaultValue: '1', type: 'integer' },
      { name: 'PH', value: '', description: 'Padding height', defaultValue: '1', type: 'integer' },
      { name: 'PW', value: '', description: 'Padding width', defaultValue: '1', type: 'integer' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'f16', type: 'dtype' },
    ],
    template: `module @conv_bn_relu {
  func.func @main(
    %input: tensor<{{B}}x{{IC}}x{{IH}}x{{IW}}x{{DTYPE}}>,
    %filter: tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>,
    %bn_scale: tensor<{{OC}}x{{DTYPE}}>,
    %bn_bias: tensor<{{OC}}x{{DTYPE}}>,
    %bn_mean: tensor<{{OC}}x{{DTYPE}}>,
    %bn_var: tensor<{{OC}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}> {
    // Step 1: 2D Convolution with padding
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %conv_init = tensor.empty() : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>
    %conv_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%conv_init : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>) -> tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>
    
    %conv = linalg.conv_2d_nchw_fchw {
      strides = dense<[{{SH}}, {{SW}}]> : tensor<2xindex>,
      dilations = dense<[1, 1]> : tensor<2xindex>
    } ins(%input, %filter : tensor<{{B}}x{{IC}}x{{IH}}x{{IW}}x{{DTYPE}}>, tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>)
      outs(%conv_filled : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>

    // Step 2: Batch Normalization (inference mode)
    // bn_out = scale * (conv - mean) / sqrt(var + eps) + bias
    %bn_result = linalg.generic {
      indexing_maps = [
        affine_map<(b, c, h, w) -> (b, c, h, w)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (c)>,
        affine_map<(b, c, h, w) -> (b, c, h, w)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%conv, %bn_scale, %bn_bias, %bn_mean, %bn_var : 
      tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>,
      tensor<{{OC}}x{{DTYPE}}>, tensor<{{OC}}x{{DTYPE}}>,
      tensor<{{OC}}x{{DTYPE}}>, tensor<{{OC}}x{{DTYPE}}>)
      outs(%conv_init : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %s: {{DTYPE}}, %b: {{DTYPE}}, %m: {{DTYPE}}, %v: {{DTYPE}}, %out: {{DTYPE}}):
      %sub = arith.subf %in, %m : {{DTYPE}}
      %rsqrt_v = math.rsqrt %v : {{DTYPE}}
      %norm = arith.mulf %sub, %rsqrt_v : {{DTYPE}}
      %scaled = arith.mulf %norm, %s : {{DTYPE}}
      %shifted = arith.addf %scaled, %b : {{DTYPE}}
      linalg.yield %shifted : {{DTYPE}}
    } -> tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>

    // Step 3: ReLU activation
    %relu_result = linalg.generic {
      indexing_maps = [
        affine_map<(b, c, h, w) -> (b, c, h, w)>,
        affine_map<(b, c, h, w) -> (b, c, h, w)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%bn_result : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>)
      outs(%conv_init : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>) {
    ^bb0(%in: {{DTYPE}}, %out: {{DTYPE}}):
      %max = arith.maximumf %in, %c0 : {{DTYPE}}
      linalg.yield %max : {{DTYPE}}
    } -> tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>

    return %relu_result : tensor<{{B}}x{{OC}}x{{IH}}x{{IW}}x{{DTYPE}}>
  }

  // Conv config: kernel={{KH}}x{{KW}}, stride={{SH}}x{{SW}}, pad={{PH}}x{{PW}}
  // Channels: {{IC}} -> {{OC}}, Batch: {{B}}, Spatial: {{IH}}x{{IW}}
}`,
  },
  {
    id: 'embedding-positional-dropout',
    name: 'Embedding + Positional Encoding + Dropout',
    category: 'Embedding',
    description: 'Token embedding lookup with sinusoidal positional encoding and dropout. Used at the input of transformer models.',
    tags: ['embedding', 'positional', 'dropout', 'transformer', 'input'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '512', type: 'dimension' },
      { name: 'V', value: '', description: 'Vocabulary size', defaultValue: '50257', type: 'dimension' },
      { name: 'D', value: '', description: 'Embedding dimension', defaultValue: '768', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'f16', type: 'dtype' },
      { name: 'DROP_RATE', value: '', description: 'Dropout rate (0.0-1.0)', defaultValue: '0.1', type: 'string' },
    ],
    template: `module @embedding_positional_dropout {
  func.func @main(
    %token_ids: tensor<{{B}}x{{S}}xindex>,
    %embed_table: tensor<{{V}}x{{D}}x{{DTYPE}}>,
    %pos_table: tensor<{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    // Step 1: Token Embedding Lookup
    %embed_result = "stablehlo.gather"(%embed_table, %token_ids) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [2],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 2,
        slice_sizes = [1, {{D}}]
      >
    } : (tensor<{{V}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{S}}xindex>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Step 2: Add Positional Encoding
    // pos_encoding[s,d] = sin(s/10000^(2k/d)) for even d
    //                    = cos(s/10000^(2k/d)) for odd d
    %pos_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    %with_pos = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, d) -> (b, s, d)>,
        affine_map<(b, s, d) -> (s, d)>,
        affine_map<(b, s, d) -> (b, s, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%embed_result, %pos_table : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{S}}x{{D}}x{{DTYPE}}>)
      outs(%pos_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) {
    ^bb0(%tok: {{DTYPE}}, %pos: {{DTYPE}}, %out: {{DTYPE}}):
      %sum = arith.addf %tok, %pos : {{DTYPE}}
      linalg.yield %sum : {{DTYPE}}
    } -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Step 3: Dropout (training mode: rate = {{DROP_RATE}})
    // At inference, dropout is identity
    %dropout_result = "stablehlo.custom_call"(%with_pos) {
      call_target_name = "dropout",
      rate = {{DROP_RATE}} : f32,
      training = false
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %dropout_result : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
  
  // Config: vocab={{V}}, seq_len={{S}}, embed_dim={{D}}, dropout={{DROP_RATE}}
  // Memory: embed_table = {{V}} * {{D}} * 2 bytes (f16)
}`,
  },
  {
    id: 'flash-attention',
    name: 'Flash Attention (Fused)',
    category: 'Attention',
    description: 'Memory-efficient fused attention using the FlashAttention algorithm. Reduces HBM reads/writes by tiling.',
    tags: ['flash-attention', 'fused', 'memory-efficient', 'transformer'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'H', value: '', description: 'Number of heads', defaultValue: '12', type: 'integer' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '2048', type: 'dimension' },
      { name: 'D', value: '', description: 'Head dimension', defaultValue: '64', type: 'dimension' },
      { name: 'BR', value: '', description: 'Block size for rows (Q tiles)', defaultValue: '128', type: 'integer' },
      { name: 'BC', value: '', description: 'Block size for columns (K,V tiles)', defaultValue: '64', type: 'integer' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'f16', type: 'dtype' },
    ],
    template: `module @flash_attention {
  // Flash Attention: Memory-efficient fused attention
  // Key insight: Compute softmax statistics in tiles to avoid materializing full SxS attention matrix
  // HBM reduction: O(S^2) -> O(S) per head
  
  func.func @flash_attn(
    %Q: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %K: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %V: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %neg_inf = arith.constant -inf : {{DTYPE}}
    %output = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
    %lse = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{DTYPE}}>  // log-sum-exp
    
    // Outer loop: tile over Q (row blocks of size {{BR}})
    // Inner loop: tile over K,V (column blocks of size {{BC}})
    %result = "stablehlo.custom_call"(%Q, %K, %V) {
      call_target_name = "flash_attention_fwd",
      // Tiling configuration
      block_rows = {{BR}} : i32,
      block_cols = {{BC}} : i32,
      // Shared memory budget: {{BR}} * {{D}} + {{BC}} * {{D}} elements
      // = ({{BR}} + {{BC}}) * {{D}} * 2 bytes = {{BR_BC_D_BYTES}} bytes per head
      causal_mask = false
    } : (tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
         tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
         tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>

    return %result : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  }
  
  // Flash Attention config:
  //   B={{B}}, H={{H}}, S={{S}}, D={{D}}
  //   Tile: BR={{BR}}, BC={{BC}}
  //   Standard attention memory: O(B*H*S*S) = O({{B}}*{{H}}*{{S}}*{{S}})
  //   Flash attention memory: O(B*H*S*D) = O({{B}}*{{H}}*{{S}}*{{D}})
  //   Memory savings: {{S}}x reduction in attention matrix storage
}`,
  },
  {
    id: 'swiglu-ffn',
    name: 'SwiGLU Feed-Forward Network',
    category: 'Linear Algebra',
    description: 'SwiGLU activation in a feed-forward network. Used in LLaMA, PaLM, and other modern LLMs.',
    tags: ['swiglu', 'ffn', 'llama', 'feed-forward', 'activation'],
    params: [
      { name: 'B', value: '', description: 'Batch size', defaultValue: '4', type: 'dimension' },
      { name: 'S', value: '', description: 'Sequence length', defaultValue: '512', type: 'dimension' },
      { name: 'D', value: '', description: 'Hidden dimension', defaultValue: '4096', type: 'dimension' },
      { name: 'D_FF', value: '', description: 'FFN intermediate dimension (typically 8/3 * D)', defaultValue: '11008', type: 'dimension' },
      { name: 'DTYPE', value: '', description: 'Data type', defaultValue: 'bf16', type: 'dtype' },
    ],
    template: `module @swiglu_ffn {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %w_gate: tensor<{{D}}x{{D_FF}}x{{DTYPE}}>,    // Gate projection
    %w_up: tensor<{{D}}x{{D_FF}}x{{DTYPE}}>,      // Up projection
    %w_down: tensor<{{D_FF}}x{{D}}x{{DTYPE}}>     // Down projection
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    
    // Step 1: Gate projection: gate = input @ w_gate
    %gate_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>
    %gate_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%gate_init : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>
    %gate = linalg.batch_matmul
      ins(%input, %w_gate : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D_FF}}x{{DTYPE}}>)
      outs(%gate_filled : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>

    // Step 2: Up projection: up = input @ w_up
    %up_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>
    %up_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%up_init : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>
    %up = linalg.batch_matmul
      ins(%input, %w_up : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D_FF}}x{{DTYPE}}>)
      outs(%up_filled : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>

    // Step 3: SwiGLU activation: swiglu(gate, up) = SiLU(gate) * up
    // SiLU(x) = x * sigmoid(x)
    %swiglu = linalg.generic {
      indexing_maps = [
        affine_map<(b, s, d) -> (b, s, d)>,
        affine_map<(b, s, d) -> (b, s, d)>,
        affine_map<(b, s, d) -> (b, s, d)>
      ],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%gate, %up : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>, tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>)
      outs(%gate_init : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>) {
    ^bb0(%g: {{DTYPE}}, %u: {{DTYPE}}, %out: {{DTYPE}}):
      %sigmoid = math.sigmoid %g : {{DTYPE}}
      %silu = arith.mulf %g, %sigmoid : {{DTYPE}}
      %result = arith.mulf %silu, %u : {{DTYPE}}
      linalg.yield %result : {{DTYPE}}
    } -> tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>

    // Step 4: Down projection: output = swiglu @ w_down
    %out_init = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %out_filled = linalg.fill ins(%c0 : {{DTYPE}}) outs(%out_init : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %output = linalg.batch_matmul
      ins(%swiglu, %w_down : tensor<{{B}}x{{S}}x{{D_FF}}x{{DTYPE}}>, tensor<{{D_FF}}x{{D}}x{{DTYPE}}>)
      outs(%out_filled : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
  
  // SwiGLU FFN config: D={{D}}, D_FF={{D_FF}}
  // Parameters: 3 * {{D}} * {{D_FF}} = {{PARAMS}} total
  // FLOPs: 2 * {{B}} * {{S}} * (2*{{D}}*{{D_FF}} + {{D_FF}}*{{D}})
}`,
  },
];

export const DEFAULT_TEMPLATE = TEMPLATES[0];
