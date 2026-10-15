import { useState, useMemo, useCallback } from 'react';

// ============================================================================
// MLIR TEMPLATE ENGINE - The actual engine logic (TypeScript port)
// This is exactly what the Rust crate does, but runs in the browser
// ============================================================================

/** Extract all {{VAR}} placeholders from a template */
function extractPlaceholders(template: string): string[] {
  const regex = /\{\{(\w+)\}\}/g;
  const found = new Set<string>();
  let match;
  while ((match = regex.exec(template)) !== null) {
    found.add(match[1]);
  }
  return Array.from(found).sort();
}

/** Count all single braces in a string (MLIR's structural braces) */
function countSingleBraces(text: string): { open: number; close: number } {
  let open = 0, close = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') {
      if (i + 1 < text.length && text[i + 1] === '{') {
        i++;
      } else {
        open++;
      }
    } else if (text[i] === '}') {
      if (i + 1 < text.length && text[i + 1] === '}') {
        i++;
      } else {
        close++;
      }
    }
  }
  return { open, close };
}

/** 
 * THE CORE ENGINE: Replace ONLY {{VAR}} placeholders.
 * Leave ALL other {} completely untouched.
 */
function renderTemplate(template: string, params: Record<string, string>): string {
  let result = template;
  for (const [key, value] of Object.entries(params)) {
    const placeholder = `{{${key}}}`;
    result = result.split(placeholder).join(value);
  }
  return result;
}

/** Simulate what Rust's format! macro would do (the BROKEN approach) */
function simulateBrokenFormat(template: string): { errors: string[] } {
  const errors: string[] = [];
  const braceMatches = template.match(/\{[^{]/g) || [];
  if (braceMatches.length > 0) {
    errors.push(`format! found ${braceMatches.length} single { that it would try to interpret as placeholders`);
  }
  return { errors };
}

// ============================================================================
// TEMPLATE DATA
// ============================================================================

interface Template {
  id: string;
  name: string;
  description: string;
  template: string;
  params: { name: string; default: string; desc: string }[];
}

const TEMPLATES: Template[] = [
  {
    id: 'matmul-bias-relu',
    name: 'MatMul + Bias + ReLU',
    description: 'Fused matrix multiply, bias add, and ReLU activation',
    params: [
      { name: 'M', default: '128', desc: 'Batch size (rows)' },
      { name: 'N', default: '256', desc: 'Output features' },
      { name: 'K', default: '512', desc: 'Input features (inner dim)' },
      { name: 'DTYPE', default: 'f16', desc: 'Data type' },
    ],
    template: `module @matmul_bias_relu {
  func.func @main(%input: tensor<{{M}}x{{K}}x{{DTYPE}}>,
                  %weight: tensor<{{K}}x{{N}}x{{DTYPE}}>,
                  %bias: tensor<{{N}}x{{DTYPE}}>) -> tensor<{{M}}x{{N}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %init = tensor.empty() : tensor<{{M}}x{{N}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%init : tensor<{{M}}x{{N}}x{{DTYPE}}>) -> tensor<{{M}}x{{N}}x{{DTYPE}}>
    
    // Matrix multiply
    %matmul = linalg.matmul
      ins(%input, %weight : tensor<{{M}}x{{K}}x{{DTYPE}}>, tensor<{{K}}x{{N}}x{{DTYPE}}>)
      outs(%filled : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>

    // Bias add (broadcast over batch dim)
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

    // ReLU: max(0, x)
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
}`,
  },
  {
    id: 'multi-head-attention',
    name: 'Multi-Head Attention',
    description: 'Scaled dot-product attention for transformers',
    params: [
      { name: 'B', default: '4', desc: 'Batch size' },
      { name: 'H', default: '8', desc: 'Number of heads' },
      { name: 'S', default: '512', desc: 'Sequence length' },
      { name: 'D', default: '64', desc: 'Head dimension' },
      { name: 'DTYPE', default: 'f16', desc: 'Data type' },
    ],
    template: `module @multi_head_attention {
  func.func @attention(
    %Q: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %K: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %V: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    
    // QK^T: attention scores
    %scores_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    %scores = linalg.batch_matmul
      ins(%Q, %K : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      outs(%scores_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>

    // Scale by 1/sqrt(d_k)
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

    // Softmax + matmul with V
    %out_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
    %output = "stablehlo.custom_call"(%scaled, %V) {
      call_target_name = "fused_attention_softmax"
    } : (tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>

    return %output : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  }
}`,
  },
  {
    id: 'layernorm-linear',
    name: 'LayerNorm + Linear',
    description: 'Pre-norm transformer block component',
    params: [
      { name: 'B', default: '4', desc: 'Batch size' },
      { name: 'S', default: '512', desc: 'Sequence length' },
      { name: 'D', default: '1024', desc: 'Hidden dimension' },
      { name: 'DTYPE', default: 'f16', desc: 'Data type' },
      { name: 'EPS', default: '1e-5', desc: 'LayerNorm epsilon' },
    ],
    template: `module @layernorm_linear {
  func.func @main(
    %input: tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>,
    %gamma: tensor<{{D}}x{{DTYPE}}>,
    %beta: tensor<{{D}}x{{DTYPE}}>,
    %weight: tensor<{{D}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}> {
    // LayerNorm: normalize over last dim
    %normalized = "stablehlo.custom_call"(%input, %gamma, %beta) {
      call_target_name = "layer_norm",
      epsilon = {{EPS}} : {{DTYPE}}
    } : (tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>, tensor<{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    // Linear projection
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %out = tensor.empty() : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    %filled = linalg.fill ins(%c0 : {{DTYPE}})
      outs(%out : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>) -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
    
    %result = linalg.batch_matmul
      ins(%normalized, %weight : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{D}}x{{D}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>

    return %result : tensor<{{B}}x{{S}}x{{D}}x{{DTYPE}}>
  }
}`,
  },
  {
    id: 'conv-bn-relu',
    name: 'Conv2D + BatchNorm + ReLU',
    description: 'CNN building block (ResNet-style)',
    params: [
      { name: 'B', default: '8', desc: 'Batch size' },
      { name: 'IC', default: '64', desc: 'Input channels' },
      { name: 'OC', default: '128', desc: 'Output channels' },
      { name: 'H', default: '56', desc: 'Spatial height' },
      { name: 'W', default: '56', desc: 'Spatial width' },
      { name: 'KH', default: '3', desc: 'Kernel height' },
      { name: 'KW', default: '3', desc: 'Kernel width' },
      { name: 'DTYPE', default: 'f16', desc: 'Data type' },
    ],
    template: `module @conv_bn_relu {
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

    // Conv2D
    %conv = linalg.conv_2d_nchw_fchw {
      strides = dense<[1, 1]> : tensor<2xindex>,
      dilations = dense<[1, 1]> : tensor<2xindex>
    } ins(%input, %filter : tensor<{{B}}x{{IC}}x{{H}}x{{W}}x{{DTYPE}}>, tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>

    // BatchNorm (inference)
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

    // ReLU
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
}`,
  },
];

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [selectedTemplate, setSelectedTemplate] = useState<Template>(TEMPLATES[0]);
  const [params, setParams] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    TEMPLATES[0].params.forEach(p => { init[p.name] = p.default; });
    return init;
  });
  const [activeTab, setActiveTab] = useState<'playground' | 'explain' | 'rust'>('playground');
  const [copied, setCopied] = useState(false);

  const placeholders = useMemo(() => extractPlaceholders(selectedTemplate.template), [selectedTemplate]);
  const rendered = useMemo(() => renderTemplate(selectedTemplate.template, params), [selectedTemplate, params]);
  const templateBraces = useMemo(() => countSingleBraces(selectedTemplate.template), [selectedTemplate]);
  const renderedBraces = useMemo(() => countSingleBraces(rendered), [rendered]);
  const brokenCheck = useMemo(() => simulateBrokenFormat(selectedTemplate.template), [selectedTemplate]);

  const handleSelectTemplate = useCallback((t: Template) => {
    setSelectedTemplate(t);
    const newParams: Record<string, string> = {};
    t.params.forEach(p => { newParams[p.name] = p.default; });
    setParams(newParams);
  }, []);

  const handleParamChange = useCallback((name: string, value: string) => {
    setParams(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(rendered);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [rendered]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                <svg className="w-6 h-6 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold">MLIR Template Toolkit</h1>
                <p className="text-xs text-gray-500">Safe interpolation — never touches MLIR's {'{ }'} braces</p>
              </div>
            </div>
            <nav className="flex gap-1 bg-gray-800/50 rounded-lg p-1">
              {(['playground', 'explain', 'rust'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                    activeTab === tab
                      ? 'bg-emerald-500/20 text-emerald-400 shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {tab === 'playground' ? '⚡ Playground' : tab === 'explain' ? '📖 How It Works' : '🦀 Rust Code'}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'playground' && (
          <PlaygroundTab
            templates={TEMPLATES}
            selectedTemplate={selectedTemplate}
            onSelectTemplate={handleSelectTemplate}
            params={params}
            onParamChange={handleParamChange}
            placeholders={placeholders}
            rendered={rendered}
            templateBraces={templateBraces}
            renderedBraces={renderedBraces}
            brokenCheck={brokenCheck}
            copied={copied}
            onCopy={handleCopy}
          />
        )}
        {activeTab === 'explain' && <ExplainTab />}
        {activeTab === 'rust' && <RustCodeTab />}
      </main>
    </div>
  );
}

// ============================================================================
// PLAYGROUND TAB
// ============================================================================

function PlaygroundTab({
  templates, selectedTemplate, onSelectTemplate, params, onParamChange,
  placeholders, rendered, templateBraces, renderedBraces, brokenCheck, copied, onCopy
}: {
  templates: Template[];
  selectedTemplate: Template;
  onSelectTemplate: (t: Template) => void;
  params: Record<string, string>;
  onParamChange: (name: string, value: string) => void;
  placeholders: string[];
  rendered: string;
  templateBraces: { open: number; close: number };
  renderedBraces: { open: number; close: number };
  brokenCheck: { errors: string[] };
  copied: boolean;
  onCopy: () => void;
}) {
  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-red-950/40 to-gray-900 border border-red-900/50 rounded-xl p-5">
        <div className="flex items-start gap-3">
          <span className="text-2xl">🔴</span>
          <div>
            <h2 className="text-lg font-bold text-red-300">The Problem</h2>
            <p className="text-sm text-gray-400 mt-1">
              Rust's <code className="px-1.5 py-0.5 bg-gray-800 rounded text-red-300">format!</code> macro 
              treats <code className="px-1.5 py-0.5 bg-gray-800 rounded text-red-300">{'{}'}</code> as placeholders. 
              MLIR has <strong className="text-white">{templateBraces.open + templateBraces.close} structural braces</strong> in this template alone. 
              Using <code className="px-1.5 py-0.5 bg-gray-800 rounded text-red-300">format!</code> would try to replace ALL of them → compile errors or corrupted output.
            </p>
          </div>
        </div>
      </div>

      <div className="bg-gradient-to-r from-emerald-950/40 to-gray-900 border border-emerald-900/50 rounded-xl p-5">
        <div className="flex items-start gap-3">
          <span className="text-2xl">🟢</span>
          <div>
            <h2 className="text-lg font-bold text-emerald-300">The Solution</h2>
            <p className="text-sm text-gray-400 mt-1">
              Use <code className="px-1.5 py-0.5 bg-gray-800 rounded text-emerald-300">{'{{VAR}}'}</code> (double braces) as placeholders. 
              The engine <strong className="text-white">ONLY replaces {'{{VAR}}'}</strong> patterns — 
              every single <code className="px-1.5 py-0.5 bg-gray-800 rounded text-emerald-300">{'{'}</code> and <code className="px-1.5 py-0.5 bg-gray-800 rounded text-emerald-300">{'}'}</code> in the MLIR stays exactly as-is.
              {brokenCheck.errors.length > 0 && (
                <span className="block mt-1 text-amber-400">
                  ⚠️ format! would fail: {brokenCheck.errors[0]}
                </span>
              )}
            </p>
          </div>
        </div>
      </div>

      <div className="flex gap-2 flex-wrap">
        {templates.map(t => (
          <button
            key={t.id}
            onClick={() => onSelectTemplate(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedTemplate.id === t.id
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600 hover:text-gray-200'
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Parameters — only these get replaced
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {selectedTemplate.params.map(p => (
            <div key={p.name}>
              <label className="flex items-center gap-1.5 mb-1">
                <code className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded">
                  {`{{${p.name}}}`}
                </code>
              </label>
              <input
                type="text"
                value={params[p.name] || ''}
                onChange={(e) => onParamChange(p.name, e.target.value)}
                placeholder={p.default}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20"
              />
              <p className="text-[10px] text-gray-600 mt-1">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-800/30">
            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
              📄 Template (input .mlir file)
            </span>
            <span className="text-xs text-gray-600">
              {placeholders.length} placeholders: {placeholders.map(p => `{{${p}}}`).join(', ')}
            </span>
          </div>
          <div className="p-4 overflow-auto max-h-[500px]">
            <pre className="text-xs font-mono leading-relaxed">
              <TemplateHighlight template={selectedTemplate.template} params={params} />
            </pre>
          </div>
        </div>

        <div className="bg-gray-900/50 border border-emerald-900/50 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-emerald-900/20">
            <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
              ✅ Rendered Output (valid MLIR)
            </span>
            <button
              onClick={onCopy}
              className="flex items-center gap-1 px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-300 transition-colors"
            >
              {copied ? '✓ Copied!' : '📋 Copy'}
            </button>
          </div>
          <div className="p-4 overflow-auto max-h-[500px]">
            <pre className="text-xs font-mono leading-relaxed">
              <RenderedHighlight rendered={rendered} params={params} />
            </pre>
          </div>
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">
          🔒 Brace Integrity Proof
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Template structural braces</p>
            <p className="text-lg font-mono font-bold text-gray-200">
              {'{'} <span className="text-emerald-400">{templateBraces.open}</span>
              {' '} {'}'} <span className="text-emerald-400">{templateBraces.close}</span>
            </p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3">
            <p className="text-xs text-gray-500 mb-1">Rendered structural braces</p>
            <p className="text-lg font-mono font-bold text-gray-200">
              {'{'} <span className="text-emerald-400">{renderedBraces.open}</span>
              {' '} {'}'} <span className="text-emerald-400">{renderedBraces.close}</span>
            </p>
          </div>
          <div className={`rounded-lg p-3 ${
            templateBraces.open === renderedBraces.open && templateBraces.close === renderedBraces.close
              ? 'bg-emerald-900/30 border border-emerald-800/50'
              : 'bg-red-900/30 border border-red-800/50'
          }`}>
            <p className="text-xs text-gray-500 mb-1">Status</p>
            <p className="text-lg font-bold text-emerald-400">
              {templateBraces.open === renderedBraces.open && templateBraces.close === renderedBraces.close
                ? '✓ All MLIR braces preserved'
                : '✗ Brace mismatch!'}
            </p>
          </div>
        </div>
        <p className="text-xs text-gray-600 mt-3">
          The engine only replaces <code className="text-emerald-400">{'{{VAR}}'}</code> patterns. 
          Every <code className="text-cyan-400">{'{'}</code> and <code className="text-cyan-400">{'}'}</code> that belongs to MLIR syntax 
          (modules, functions, affine maps, tensor types, linalg regions) is completely untouched.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// EXPLAIN TAB
// ============================================================================

function ExplainTab() {
  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h2 className="text-2xl font-bold mb-4">How the Template Engine Works</h2>
        <p className="text-gray-400 leading-relaxed">
          The core insight is simple: <strong className="text-white">use a placeholder syntax that cannot conflict with MLIR</strong>.
          MLIR uses single braces <code className="px-1.5 py-0.5 bg-gray-800 rounded text-cyan-400">{'{'}</code> <code className="px-1.5 py-0.5 bg-gray-800 rounded text-cyan-400">{'}'}</code> everywhere.
          So we use <strong className="text-emerald-400">double braces</strong> <code className="px-1.5 py-0.5 bg-gray-800 rounded text-emerald-400">{'{{VAR}}'}</code> for our placeholders.
        </p>
      </div>

      <div className="space-y-4">
        <h3 className="text-xl font-bold">The Algorithm (3 steps)</h3>
        
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-emerald-400">1</span>
            </div>
            <div>
              <h4 className="font-semibold text-gray-200">Scan for {'{{VAR}}'} patterns</h4>
              <p className="text-sm text-gray-400 mt-1">
                Use regex <code className="text-emerald-400">{'\\{\\{(\\w+)\\}\\}'}</code> to find all placeholders. 
                This regex matches <em>exactly</em> two opening braces, a word, and two closing braces. 
                Single braces in MLIR (like <code className="text-cyan-400">{'func.func @main() {'}</code>) do NOT match.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-emerald-400">2</span>
            </div>
            <div>
              <h4 className="font-semibold text-gray-200">Replace ONLY those patterns</h4>
              <p className="text-sm text-gray-400 mt-1">
                For each parameter, do a simple string replace: <code className="text-emerald-400">{'{{M}}'}</code> → <code className="text-white">128</code>. 
                This uses <code className="text-cyan-400">str::replace()</code> (not <code className="text-red-400">format!</code>). 
                The engine never sees or touches any single brace.
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
          <div className="flex items-start gap-4">
            <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
              <span className="text-sm font-bold text-emerald-400">3</span>
            </div>
            <div>
              <h4 className="font-semibold text-gray-200">Validate output</h4>
              <p className="text-sm text-gray-400 mt-1">
                Check that no unresolved <code className="text-red-400">{'{{...}}'}</code> remain, 
                and that all MLIR structural braces are balanced. This runs at <strong>compile time</strong> via <code className="text-cyan-400">build.rs</code>.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <h3 className="text-xl font-bold mb-4">Why Double Braces?</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2 px-3 text-gray-400">Syntax</th>
                <th className="text-left py-2 px-3 text-gray-400">Used by</th>
                <th className="text-left py-2 px-3 text-gray-400">Conflict?</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-gray-800">
                <td className="py-2 px-3 font-mono text-red-400">{'{var}'}</td>
                <td className="py-2 px-3 text-gray-400">Rust format!, Python f-strings</td>
                <td className="py-2 px-3 text-red-400">❌ Conflicts with MLIR</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="py-2 px-3 font-mono text-red-400">{'${var}'}</td>
                <td className="py-2 px-3 text-gray-400">Shell, JS template literals</td>
                <td className="py-2 px-3 text-amber-400">⚠️ Could conflict with MLIR attributes</td>
              </tr>
              <tr className="border-b border-gray-800">
                <td className="py-2 px-3 font-mono text-red-400">{'<%var%>'}</td>
                <td className="py-2 px-3 text-gray-400">ERB, old ASP</td>
                <td className="py-2 px-3 text-amber-400">⚠️ Conflicts with tensor angle brackets</td>
              </tr>
              <tr className="bg-emerald-900/10">
                <td className="py-2 px-3 font-mono text-emerald-400">{'{{VAR}}'}</td>
                <td className="py-2 px-3 text-gray-400">Handlebars, Jinja2, Mustache</td>
                <td className="py-2 px-3 text-emerald-400">✅ No conflict with MLIR!</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-sm text-gray-500 mt-3">
          MLIR never uses <code className="text-emerald-400">{'{{'}</code> or <code className="text-emerald-400">{'}}'}</code> in its syntax. 
          The double-brace convention (from Handlebars/Jinja) is the proven solution.
        </p>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <h3 className="text-xl font-bold mb-4">Zero format! Usage</h3>
        <p className="text-sm text-gray-400 mb-3">
          The entire engine uses <strong className="text-white">string concatenation and replace</strong> only:
        </p>
        <pre className="bg-gray-800/50 rounded-lg p-4 text-xs font-mono text-gray-300 overflow-x-auto">
{`// Rust implementation (simplified)
fn render(&self, params: &HashMap<String, String>) -> String {
    let mut result = self.template.clone();
    for (key, value) in params {
        let placeholder = format!("{{{{{{}}}}}}", key);  // builds "{{KEY}}"
        result = result.replace(&placeholder, value);    // simple string replace
    }
    result  // All MLIR {} untouched!
}`}
        </pre>
        <p className="text-xs text-gray-600 mt-2">
          Note: The only <code>format!</code> call is to BUILD the placeholder string <code className="text-emerald-400">{'"{{KEY}}"'}</code> — 
          it's never used on the MLIR template itself.
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// RUST CODE TAB
// ============================================================================

function RustCodeTab() {
  const [copiedFile, setCopiedFile] = useState<string | null>(null);

  const copyCode = async (id: string, code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedFile(id);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold mb-2">Rust Crate Implementation</h2>
        <p className="text-gray-400">Complete, correct, compilable code. Copy these files into your project.</p>
      </div>

      <CodeBlock id="cargo" title="Cargo.toml" copied={copiedFile === 'cargo'} onCopy={() => copyCode('cargo', CARGO_TOML)} code={CARGO_TOML} />
      <CodeBlock id="lib" title="src/lib.rs" copied={copiedFile === 'lib'} onCopy={() => copyCode('lib', LIB_RS)} code={LIB_RS} />
      <CodeBlock id="render" title="src/render.rs" copied={copiedFile === 'render'} onCopy={() => copyCode('render', RENDER_RS)} code={RENDER_RS} />
      <CodeBlock id="validate" title="src/validate.rs" copied={copiedFile === 'validate'} onCopy={() => copyCode('validate', VALIDATE_RS)} code={VALIDATE_RS} />
      <CodeBlock id="build" title="build.rs" copied={copiedFile === 'build'} onCopy={() => copyCode('build', BUILD_RS)} code={BUILD_RS} />
      <CodeBlock id="usage" title="Usage Example (in your neural_flow/mod.rs)" copied={copiedFile === 'usage'} onCopy={() => copyCode('usage', USAGE_RS)} code={USAGE_RS} />
    </div>
  );
}

function CodeBlock({ title, code, copied, onCopy }: {
  id: string; title: string; code: string; copied: boolean; onCopy: () => void;
}) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-800/30">
        <span className="text-sm font-medium text-gray-300">{title}</span>
        <button
          onClick={onCopy}
          className="flex items-center gap-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-300 transition-colors"
        >
          {copied ? '✓ Copied!' : '📋 Copy'}
        </button>
      </div>
      <pre className="p-4 text-xs font-mono text-gray-300 overflow-x-auto leading-relaxed max-h-[500px] overflow-y-auto">
        {code}
      </pre>
    </div>
  );
}

// ============================================================================
// SYNTAX HIGHLIGHTING COMPONENTS
// ============================================================================

function TemplateHighlight({ template, params }: { template: string; params: Record<string, string> }) {
  const lines = template.split('\n');
  return (
    <>
      {lines.map((line, i) => (
        <div key={i} className="flex hover:bg-gray-800/30">
          <span className="inline-block w-8 text-right mr-4 text-gray-700 select-none text-[10px] leading-relaxed flex-shrink-0">
            {i + 1}
          </span>
          <span className="flex-1">
            <HighlightLine line={line} params={params} mode="template" />
          </span>
        </div>
      ))}
    </>
  );
}

function RenderedHighlight({ rendered, params }: { rendered: string; params: Record<string, string> }) {
  const lines = rendered.split('\n');
  return (
    <>
      {lines.map((line, i) => (
        <div key={i} className="flex hover:bg-emerald-900/10">
          <span className="inline-block w-8 text-right mr-4 text-gray-700 select-none text-[10px] leading-relaxed flex-shrink-0">
            {i + 1}
          </span>
          <span className="flex-1">
            <HighlightLine line={line} params={params} mode="rendered" />
          </span>
        </div>
      ))}
    </>
  );
}

function HighlightLine({ line, params, mode }: { line: string; params: Record<string, string>; mode: 'template' | 'rendered' }) {
  if (mode === 'template') {
    const parts = line.split(/(\{\{\w+\}\})/g);
    return (
      <>
        {parts.map((part, i) => {
          if (part.match(/^\{\{\w+\}\}$/)) {
            const varName = part.slice(2, -2);
            const hasValue = params[varName] !== undefined && params[varName] !== '';
            return (
              <span
                key={i}
                className={`px-0.5 rounded ${
                  hasValue
                    ? 'bg-emerald-500/30 text-emerald-300 border-b-2 border-emerald-400'
                    : 'bg-red-500/30 text-red-300 border-b-2 border-red-400 animate-pulse'
                }`}
              >
                {part}
              </span>
            );
          }
          return <span key={i}>{colorizeMlir(part)}</span>;
        })}
      </>
    );
  } else {
    const values = Object.values(params).filter(v => v.length > 0);
    if (values.length === 0) return <span>{colorizeMlir(line)}</span>;
    
    const escapedValues = values.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    const regex = new RegExp(`(${escapedValues.join('|')})`, 'g');
    const parts = line.split(regex);
    
    return (
      <>
        {parts.map((part, i) => {
          if (values.includes(part)) {
            return (
              <span key={i} className="bg-emerald-500/20 text-emerald-300 px-0.5 rounded border-b border-emerald-500/40">
                {part}
              </span>
            );
          }
          return <span key={i}>{colorizeMlir(part)}</span>;
        })}
      </>
    );
  }
}

function colorizeMlir(text: string): React.ReactNode {
  const parts = text.split(/(\b(?:module|func\.func|func|return|tensor|affine_map|iterator_types|indexing_maps|ins|outs|dense|arith\.|linalg\.|math\.|stablehlo\.|scf\.|memref\.|vector\.|cf\.)\w*|%\w+|"[^"]*"|\/\/.*$)/g);
  
  return (
    <>
      {parts.map((part, i) => {
        if (!part) return null;
        if (part.startsWith('//')) return <span key={i} className="text-gray-600 italic">{part}</span>;
        if (part.startsWith('"')) return <span key={i} className="text-amber-300">{part}</span>;
        if (part.startsWith('%')) return <span key={i} className="text-sky-300">{part}</span>;
        if (/^(module|func\.func|func|return|affine_map|iterator_types|indexing_maps|ins|outs|dense)$/.test(part))
          return <span key={i} className="text-purple-400">{part}</span>;
        if (/^(arith|linalg|math|stablehlo|scf|memref|vector|cf)\./.test(part))
          return <span key={i} className="text-cyan-400">{part}</span>;
        if (/^(f16|f32|f64|bf16|i1|i8|i16|i32|i64|index)$/.test(part))
          return <span key={i} className="text-emerald-400">{part}</span>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ============================================================================
// RUST CODE STRINGS
// ============================================================================

const CARGO_TOML = `[package]
name = "mlir-templates"
version = "0.1.0"
edition = "2021"
description = "Safe MLIR template engine - {{VAR}} interpolation that never touches MLIR braces"
license = "MIT OR Apache-2.0"

[dependencies]
thiserror = "1.0"
regex = "1.10"
once_cell = "1.19"

[build-dependencies]
regex = "1.10"`;

const LIB_RS = `//! MLIR Template Engine
//!
//! Safe string interpolation for MLIR templates using {{VAR}} syntax.
//! NEVER uses format! on MLIR content - only string replace.

pub mod render;
pub mod validate;

pub use render::MlirTemplate;
pub use validate::{validate_template, ValidationError};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum TemplateError {
    #[error("Template not found: {0}")]
    NotFound(String),
    
    #[error("Missing required parameter: {{{{ {0} }}}}")]
    MissingParameter(String),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, TemplateError>;`;

const RENDER_RS = `use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use regex::Regex;
use once_cell::sync::Lazy;

use crate::{TemplateError, Result};

/// Regex to match {{VAR}} placeholders (double braces only)
static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

/// MLIR Template with safe {{VAR}} interpolation
///
/// # Key Design
/// - Uses {{VAR}} (double braces) as placeholders
/// - MLIR's single braces {} are NEVER touched
/// - No format! macro usage on template content
/// - Simple string::replace() for substitution
pub struct MlirTemplate {
    template: String,
    required_params: HashSet<String>,
    name: String,
}

impl MlirTemplate {
    pub fn new(template: impl Into<String>) -> Result<Self> {
        let template = template.into();
        let required_params = Self::extract_params(&template);
        
        Ok(Self {
            template,
            required_params,
            name: "<inline>".to_string(),
        })
    }
    
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let name = path.as_ref()
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("<file>")
            .to_string();
        
        let required_params = Self::extract_params(&content);
        
        Ok(Self {
            template: content,
            required_params,
            name,
        })
    }
    
    pub fn from_embedded(name: &str, content: &'static str) -> Result<Self> {
        let required_params = Self::extract_params(content);
        
        Ok(Self {
            template: content.to_string(),
            required_params,
            name: name.to_string(),
        })
    }
    
    fn extract_params(template: &str) -> HashSet<String> {
        PLACEHOLDER_RE
            .captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
            .collect()
    }
    
    pub fn required_params(&self) -> &HashSet<String> {
        &self.required_params
    }
    
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Render the template with given parameters
    ///
    /// # Important
    /// This ONLY replaces {{VAR}} patterns.
    /// ALL other braces (MLIR's structural {}) are untouched.
    pub fn render(&self, params: &HashMap<String, String>) -> Result<String> {
        for param in &self.required_params {
            if !params.contains_key(param) {
                return Err(TemplateError::MissingParameter(param.clone()));
            }
        }
        
        // Simple string replacement - NO format! usage
        let mut result = self.template.clone();
        for (key, value) in params {
            let placeholder = Self::make_placeholder(key);
            result = result.replace(&placeholder, value);
        }
        
        Ok(result)
    }
    
    /// Build a {{VAR}} placeholder string using concatenation, NOT format!
    fn make_placeholder(name: &str) -> String {
        let mut s = String::with_capacity(name.len() + 4);
        s.push('{');
        s.push('{');
        s.push_str(name);
        s.push('}');
        s.push('}');
        s
    }
    
    pub fn render_with_defaults(
        &self,
        params: &HashMap<String, String>,
        defaults: &HashMap<String, String>,
    ) -> Result<String> {
        let mut merged = defaults.clone();
        merged.extend(params.clone());
        self.render(&merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_render_replaces_only_double_braces() {
        let template = r#"module @test {
  func.func @main() -> tensor<{{M}}x{{N}}xf16> {
    return %0 : tensor<{{M}}x{{N}}xf16>
  }
}"#;
        let tmpl = MlirTemplate::new(template).unwrap();
        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());
        params.insert("N".to_string(), "256".to_string());
        
        let result = tmpl.render(&params).unwrap();
        
        assert!(result.contains("128"));
        assert!(result.contains("256"));
        assert!(!result.contains("{{M}}"));
        assert!(!result.contains("{{N}}"));
        
        // MLIR's single braces should be PRESERVED
        assert!(result.contains("module @test {"));
        assert!(result.contains("func.func @main()"));
        assert!(result.contains("}"));
    }
    
    #[test]
    fn test_missing_param_errors() {
        let tmpl = MlirTemplate::new("test {{M}} {{N}}").unwrap();
        let mut params = HashMap::new();
        params.insert("M".to_string(), "128".to_string());
        
        let result = tmpl.render(&params);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_mlir_braces_preserved() {
        let template = r#"module @test {
  func.func @f() {
    %0 = arith.constant 0.0 : f32
    return
  }
}"#;
        let tmpl = MlirTemplate::new(template).unwrap();
        let params = HashMap::new();
        let result = tmpl.render(&params).unwrap();
        
        let template_open = template.matches('{').count();
        let result_open = result.matches('{').count();
        assert_eq!(template_open, result_open, "Opening braces changed!");
        
        let template_close = template.matches('}').count();
        let result_close = result.matches('}').count();
        assert_eq!(template_close, result_close, "Closing braces changed!");
    }
}`;

const VALIDATE_RS = `use std::collections::HashSet;
use regex::Regex;
use once_cell::sync::Lazy;

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

#[derive(Debug)]
pub struct ValidationError {
    pub message: String,
    pub line: Option<usize>,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(line) = self.line {
            write!(f, "line {}: {}", line, self.message)
        } else {
            write!(f, "{}", self.message)
        }
    }
}

/// Validate a template at build time
pub fn validate_template(template: &str) -> std::result::Result<(), Vec<ValidationError>> {
    let mut errors = Vec::new();
    
    // Check balanced MLIR braces
    let mut depth = 0i32;
    let mut chars = template.chars().peekable();
    let mut line = 1;
    
    while let Some(ch) = chars.next() {
        match ch {
            '\\n' => line += 1,
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next(); // skip second { (placeholder)
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next(); // skip second } (placeholder)
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push(ValidationError {
                            message: "Unbalanced brace: extra '}'".to_string(),
                            line: Some(line),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    
    if depth > 0 {
        errors.push(ValidationError {
            message: format!("Unbalanced braces: {} unclosed '{{'", depth),
            line: None,
        });
    }
    
    // Check valid placeholder names
    for cap in PLACEHOLDER_RE.captures_iter(template) {
        let name = cap.get(1).unwrap().as_str();
        if name.is_empty() {
            errors.push(ValidationError {
                message: "Empty placeholder name: {{}}".to_string(),
                line: None,
            });
        }
        if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
            errors.push(ValidationError {
                message: format!("Placeholder name cannot start with digit: {{{{{}}}}}", name),
                line: None,
            });
        }
    }
    
    // Check no nested placeholders
    if template.contains("{{{{") || template.contains("}}}}") {
        errors.push(ValidationError {
            message: "Nested placeholders detected (quadruple braces)".to_string(),
            line: None,
        });
    }
    
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

pub fn validate_and_extract(template: &str) -> std::result::Result<HashSet<String>, Vec<ValidationError>> {
    validate_template(template)?;
    
    let params: HashSet<String> = PLACEHOLDER_RE
        .captures_iter(template)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().to_string()))
        .collect();
    
    Ok(params)
}`;

const BUILD_RS = `use std::fs;
use std::path::Path;

fn main() {
    let templates_dir = Path::new("templates");
    
    if !templates_dir.exists() {
        println!("cargo:warning=No templates/ directory found, skipping validation");
        return;
    }
    
    let mut has_errors = false;
    
    for entry in fs::read_dir(templates_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        
        if path.extension().map_or(false, |ext| ext == "mlir") {
            let content = fs::read_to_string(&path).unwrap();
            let name = path.file_stem().unwrap().to_str().unwrap();
            
            match validate_template(&content) {
                Ok(()) => {
                    println!("cargo:warning=✓ Validated: {}.mlir", name);
                }
                Err(errors) => {
                    has_errors = true;
                    for err in &errors {
                        eprintln!("✗ {}.mlir: {}", name, err);
                    }
                }
            }
        }
    }
    
    if has_errors {
        panic!("Template validation failed! Fix the errors above.");
    }
    
    println!("cargo:rerun-if-changed=templates/");
}

fn validate_template(template: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut depth = 0i32;
    let mut chars = template.chars().peekable();
    
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next();
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next();
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push("Unbalanced brace: extra '}'".to_string());
                    }
                }
            }
            _ => {}
        }
    }
    
    if depth > 0 {
        errors.push(format!("Unbalanced braces: {} unclosed", depth));
    }
    
    if template.contains("{{{{") {
        errors.push("Nested placeholders detected".to_string());
    }
    
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}`;

const USAGE_RS = `// In your neural_flow/mod.rs or wherever you generate MLIR:

use mlir_templates::MlirTemplate;
use std::collections::HashMap;

// Option 1: Load from embedded template (recommended)
fn generate_matmul_bias_relu(m: usize, n: usize, k: usize) -> mlir_templates::Result<String> {
    let template = MlirTemplate::from_embedded(
        "matmul_bias_relu",
        include_str!("../templates/matmul_bias_relu.mlir")
    )?;
    
    let mut params = HashMap::new();
    params.insert("M".to_string(), m.to_string());
    params.insert("N".to_string(), n.to_string());
    params.insert("K".to_string(), k.to_string());
    params.insert("DTYPE".to_string(), "f16".to_string());
    
    // This ONLY replaces {{M}}, {{N}}, {{K}}, {{DTYPE}}
    // All MLIR braces {} are completely untouched
    template.render(&params)
}

// Option 2: Load from file at runtime
fn generate_from_file(template_path: &str, params: &HashMap<String, String>) -> mlir_templates::Result<String> {
    let template = MlirTemplate::from_file(template_path)?;
    template.render(params)
}

// Option 3: Inline template string
fn generate_custom() -> mlir_templates::Result<String> {
    let template_str = r#"
module @custom_op {
  func.func @main(%arg0: tensor<{{DIM}}x{{DIM}}x{{DTYPE}}>) -> tensor<{{DIM}}x{{DIM}}x{{DTYPE}}> {
    return %arg0 : tensor<{{DIM}}x{{DIM}}x{{DTYPE}}>
  }
}
"#;
    let template = MlirTemplate::new(template_str)?;
    let mut params = HashMap::new();
    params.insert("DIM".to_string(), "256".to_string());
    params.insert("DTYPE".to_string(), "f32".to_string());
    template.render(&params)
}

// ============================================================================
// REPLACING YOUR OLD CODE:
// ============================================================================
//
// BEFORE (broken - format! eats MLIR braces):
//   format!(r###"module @test {{ func @main() -> tensor<{}x{}xf16> {{ ... }} }}"###, m, n)
//   ^^^ BROKEN: Rust tries to interpret ALL {} as format placeholders
//
// AFTER (fixed - template engine only touches {{VAR}}):
//   let template = MlirTemplate::from_file("templates/test.mlir")?;
//   let result = template.render(&hashmap!{"M" => m.to_string(), "N" => n.to_string()})?;
//   ^^^ FIXED: Only {{M}} and {{N}} are replaced, all other {} preserved
//
// ============================================================================
//
// For your FusedOp enum, replace the generate_mlir_for_op function:
//
// BEFORE:
//   FusedOp::MatMulBiasRelu { m, n, k } => {
//       Ok(format!("module @matmul {{ ... tensor<{}x{}> ... }}", m, n))
//   }
//
// AFTER:
//   FusedOp::MatMulBiasRelu { m, n, k } => {
//       let template = MlirTemplate::from_embedded("matmul_bias_relu", MATMUL_TEMPLATE)?;
//       let mut params = HashMap::new();
//       params.insert("M".into(), m.to_string());
//       params.insert("N".into(), n.to_string());
//       params.insert("K".into(), k.to_string());
//       template.render(&params)
//   }`;
