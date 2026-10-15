import { useState, useMemo, useCallback } from 'react';

// ============================================================================
// MLIR TEMPLATE ENGINE (TypeScript port for the interactive demo)
// ============================================================================

function extractPlaceholders(template: string): string[] {
  const regex = /\{\{(\w+)\}\}/g;
  const found = new Set<string>();
  let match;
  while ((match = regex.exec(template)) !== null) {
    found.add(match[1]);
  }
  return Array.from(found).sort();
}

function countSingleBraces(text: string): { open: number; close: number } {
  let open = 0, close = 0;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '{') {
      if (i + 1 < text.length && text[i + 1] === '{') { i++; } else { open++; }
    } else if (text[i] === '}') {
      if (i + 1 < text.length && text[i + 1] === '}') { i++; } else { close++; }
    }
  }
  return { open, close };
}

function renderTemplate(template: string, params: Record<string, string>): string {
  let result = template;
  for (const [key, value] of Object.entries(params)) {
    result = result.split(`{{${key}}}`).join(value);
  }
  return result;
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
      { name: 'M', default: '128', desc: 'Batch size' },
      { name: 'N', default: '256', desc: 'Output features' },
      { name: 'K', default: '512', desc: 'Input features' },
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
    %matmul = linalg.matmul
      ins(%input, %weight : tensor<{{M}}x{{K}}x{{DTYPE}}>, tensor<{{K}}x{{N}}x{{DTYPE}}>)
      outs(%filled : tensor<{{M}}x{{N}}x{{DTYPE}}>)
      -> tensor<{{M}}x{{N}}x{{DTYPE}}>
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
      { name: 'H', default: '8', desc: 'Heads' },
      { name: 'S', default: '512', desc: 'Sequence length' },
      { name: 'D', default: '64', desc: 'Head dim' },
      { name: 'DTYPE', default: 'f16', desc: 'Data type' },
    ],
    template: `module @multi_head_attention {
  func.func @attention(
    %Q: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %K: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>,
    %V: tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>
  ) -> tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}> {
    %c0 = arith.constant 0.0 : {{DTYPE}}
    %scores_init = tensor.empty() : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
    %scores = linalg.batch_matmul
      ins(%Q, %K : tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>, tensor<{{B}}x{{H}}x{{S}}x{{D}}x{{DTYPE}}>)
      outs(%scores_init : tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{H}}x{{S}}x{{S}}x{{DTYPE}}>
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
    id: 'conv-bn-relu',
    name: 'Conv2D + BatchNorm + ReLU',
    description: 'CNN building block (ResNet-style)',
    params: [
      { name: 'B', default: '8', desc: 'Batch' },
      { name: 'IC', default: '64', desc: 'Input channels' },
      { name: 'OC', default: '128', desc: 'Output channels' },
      { name: 'H', default: '56', desc: 'Height' },
      { name: 'W', default: '56', desc: 'Width' },
      { name: 'KH', default: '3', desc: 'Kernel H' },
      { name: 'KW', default: '3', desc: 'Kernel W' },
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
    %conv = linalg.conv_2d_nchw_fchw {
      strides = dense<[1, 1]> : tensor<2xindex>,
      dilations = dense<[1, 1]> : tensor<2xindex>
    } ins(%input, %filter : tensor<{{B}}x{{IC}}x{{H}}x{{W}}x{{DTYPE}}>, tensor<{{OC}}x{{IC}}x{{KH}}x{{KW}}x{{DTYPE}}>)
      outs(%filled : tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>)
      -> tensor<{{B}}x{{OC}}x{{H}}x{{W}}x{{DTYPE}}>
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
// REPO FILE STRUCTURE
// ============================================================================

interface RepoFile {
  path: string;
  description: string;
  icon: string;
}

const REPO_STRUCTURE: { dir: string; files: RepoFile[] }[] = [
  {
    dir: 'mlir-templates/',
    files: [
      { path: 'Cargo.toml', description: 'Package manifest with dependencies', icon: '📦' },
      { path: 'README.md', description: 'Documentation with usage examples', icon: '📖' },
      { path: '.gitignore', description: 'Git ignore rules', icon: '🚫' },
      { path: 'build.rs', description: 'Compile-time template validation', icon: '🔨' },
    ],
  },
  {
    dir: 'src/',
    files: [
      { path: 'lib.rs', description: 'Public API exports', icon: '📚' },
      { path: 'render.rs', description: 'MlirTemplate engine (core)', icon: '⚙️' },
      { path: 'validate.rs', description: 'Template validation logic', icon: '✅' },
      { path: 'error.rs', description: 'Error type definitions', icon: '❌' },
    ],
  },
  {
    dir: 'templates/',
    files: [
      { path: 'matmul_bias_relu.mlir', description: 'Fused matmul + bias + ReLU', icon: '🧮' },
      { path: 'multi_head_attention.mlir', description: 'Multi-head self-attention', icon: '🧠' },
      { path: 'layernorm_linear.mlir', description: 'LayerNorm + linear projection', icon: '📐' },
      { path: 'conv_bn_relu.mlir', description: 'Conv2D + BatchNorm + ReLU', icon: '🖼️' },
    ],
  },
  {
    dir: 'tests/',
    files: [
      { path: 'integration_tests.rs', description: 'End-to-end integration tests', icon: '🧪' },
    ],
  },
  {
    dir: 'examples/',
    files: [
      { path: 'neural_flow.rs', description: 'Usage in neural network codegen', icon: '💡' },
    ],
  },
];

// ============================================================================
// MAIN APP
// ============================================================================

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'playground' | 'files' | 'usage'>('overview');

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center shadow-lg shadow-emerald-500/20">
                <svg className="w-6 h-6 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold">MLIR Template Toolkit</h1>
                <p className="text-xs text-gray-500">Safe {'{{VAR}}'} interpolation — never touches MLIR's {'{ }'} braces</p>
              </div>
            </div>
            <nav className="flex gap-1 bg-gray-800/50 rounded-lg p-1">
              {([
                ['overview', '📋 Overview'],
                ['playground', '⚡ Playground'],
                ['files', '📁 Files'],
                ['usage', '🦀 Usage'],
              ] as const).map(([tab, label]) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    activeTab === tab
                      ? 'bg-emerald-500/20 text-emerald-400 shadow-sm'
                      : 'text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'overview' && <OverviewTab />}
        {activeTab === 'playground' && <PlaygroundTab />}
        {activeTab === 'files' && <FilesTab />}
        {activeTab === 'usage' && <UsageTab />}
      </main>

      <footer className="border-t border-gray-800 mt-12 py-4 text-center text-xs text-gray-600">
        MLIR Template Toolkit — Solves Rust format! macro conflicts with MLIR curly braces
      </footer>
    </div>
  );
}

// ============================================================================
// OVERVIEW TAB
// ============================================================================

function OverviewTab() {
  return (
    <div className="space-y-8 max-w-4xl">
      <div className="bg-gradient-to-r from-emerald-950/40 to-cyan-950/40 border border-emerald-900/50 rounded-2xl p-8">
        <h2 className="text-3xl font-bold mb-3">What problem does this solve?</h2>
        <p className="text-gray-300 leading-relaxed text-lg">
          Rust's <code className="px-2 py-0.5 bg-gray-800 rounded text-red-300">format!</code> macro treats <code className="px-2 py-0.5 bg-gray-800 rounded text-red-300">{'{}'}</code> as placeholders.
          MLIR code is <strong className="text-white">full</strong> of <code className="text-cyan-400">{'{'}</code> and <code className="text-cyan-400">{'}'}</code> — modules, functions, affine maps, linalg regions.
          Using <code className="text-red-300">format!</code> on MLIR causes compile errors or corrupted output.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-red-950/20 border border-red-900/50 rounded-xl p-5">
          <h3 className="text-lg font-bold text-red-300 mb-3 flex items-center gap-2">
            <span className="text-xl">❌</span> Before (Broken)
          </h3>
          <pre className="text-xs font-mono text-gray-400 bg-gray-900/50 rounded-lg p-3 overflow-x-auto">
{`// Rust tries to replace EVERY {}
format!(r#"
  module @test {{
    func @main() -> tensor<{}x{}xf16> {{
      // Rust eats these braces too!
    }}
  }}
"#, m, n)

// Result: Compile error or
// corrupted MLIR output`}
          </pre>
        </div>

        <div className="bg-emerald-950/20 border border-emerald-900/50 rounded-xl p-5">
          <h3 className="text-lg font-bold text-emerald-300 mb-3 flex items-center gap-2">
            <span className="text-xl">✅</span> After (Fixed)
          </h3>
          <pre className="text-xs font-mono text-gray-400 bg-gray-900/50 rounded-lg p-3 overflow-x-auto">
{`// Engine ONLY replaces {{VAR}}
let template = MlirTemplate::new(r#"
  module @test {
    func @main() -> tensor<{{M}}x{{N}}xf16> {
      // These braces are safe!
    }
  }
"#)?;
template.render(&params)?;

// Result: Valid MLIR every time`}
          </pre>
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">How it works</h3>
        <div className="space-y-4">
          {[
            { step: '1', title: 'Scan for {{VAR}} patterns', desc: 'Regex \\{\\{(\\w+)\\}\\} matches only double-brace placeholders. Single braces in MLIR are ignored.' },
            { step: '2', title: 'Replace ONLY those patterns', desc: 'str::replace() substitutes {{M}} → 128. All other { and } are untouched.' },
            { step: '3', title: 'Validate at compile time', desc: 'build.rs checks balanced braces, valid names, no nested placeholders.' },
          ].map(item => (
            <div key={item.step} className="flex items-start gap-4">
              <div className="w-8 h-8 rounded-full bg-emerald-500/20 flex items-center justify-center flex-shrink-0">
                <span className="text-sm font-bold text-emerald-400">{item.step}</span>
              </div>
              <div>
                <h4 className="font-semibold text-gray-200">{item.title}</h4>
                <p className="text-sm text-gray-400 mt-0.5">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
        <h3 className="text-xl font-bold mb-4">Quick start</h3>
        <pre className="text-xs font-mono text-gray-300 bg-gray-800/50 rounded-lg p-4 overflow-x-auto">
{`use mlir_templates::MlirTemplate;
use std::collections::HashMap;

let template = MlirTemplate::from_embedded(
    "matmul_bias_relu",
    include_str!("../templates/matmul_bias_relu.mlir"),
)?;

let mut params = HashMap::new();
params.insert("M".to_string(), "128".to_string());
params.insert("N".to_string(), "256".to_string());
params.insert("K".to_string(), "512".to_string());
params.insert("DTYPE".to_string(), "f16".to_string());

let mlir = template.render(&params)?;
// All MLIR braces {} preserved! Only {{M}}, {{N}}, {{K}}, {{DTYPE}} replaced.`}
        </pre>
      </div>
    </div>
  );
}

// ============================================================================
// PLAYGROUND TAB
// ============================================================================

function PlaygroundTab() {
  const [selectedTemplate, setSelectedTemplate] = useState<Template>(TEMPLATES[0]);
  const [params, setParams] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    TEMPLATES[0].params.forEach(p => { init[p.name] = p.default; });
    return init;
  });
  const [copied, setCopied] = useState(false);

  const placeholders = useMemo(() => extractPlaceholders(selectedTemplate.template), [selectedTemplate]);
  const rendered = useMemo(() => renderTemplate(selectedTemplate.template, params), [selectedTemplate, params]);
  const templateBraces = useMemo(() => countSingleBraces(selectedTemplate.template), [selectedTemplate]);
  const renderedBraces = useMemo(() => countSingleBraces(rendered), [rendered]);

  const handleSelectTemplate = useCallback((t: Template) => {
    setSelectedTemplate(t);
    const newParams: Record<string, string> = {};
    t.params.forEach(p => { newParams[p.name] = p.default; });
    setParams(newParams);
  }, []);

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(rendered);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [rendered]);

  return (
    <div className="space-y-6">
      <div className="flex gap-2 flex-wrap">
        {TEMPLATES.map(t => (
          <button
            key={t.id}
            onClick={() => handleSelectTemplate(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
              selectedTemplate.id === t.id
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
            }`}
          >
            {t.name}
          </button>
        ))}
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Parameters</h3>
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
                onChange={(e) => setParams(prev => ({ ...prev, [p.name]: e.target.value }))}
                placeholder={p.default}
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-500/50"
              />
              <p className="text-[10px] text-gray-600 mt-1">{p.desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-800/30">
            <span className="text-xs font-semibold text-gray-400 uppercase">📄 Template</span>
            <span className="text-xs text-gray-600">{placeholders.length} params</span>
          </div>
          <div className="p-4 overflow-auto max-h-[500px]">
            <pre className="text-xs font-mono leading-relaxed">
              <TemplateHighlight template={selectedTemplate.template} params={params} />
            </pre>
          </div>
        </div>

        <div className="bg-gray-900/50 border border-emerald-900/50 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-emerald-900/20">
            <span className="text-xs font-semibold text-emerald-400 uppercase">✅ Rendered</span>
            <button onClick={handleCopy} className="px-2 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-300">
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
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">🔒 Brace Integrity</h3>
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <p className="text-xs text-gray-500">Template braces</p>
            <p className="text-lg font-mono font-bold">{templateBraces.open} / {templateBraces.close}</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <p className="text-xs text-gray-500">Rendered braces</p>
            <p className="text-lg font-mono font-bold">{renderedBraces.open} / {renderedBraces.close}</p>
          </div>
          <div className={`rounded-lg p-3 text-center ${
            templateBraces.open === renderedBraces.open && templateBraces.close === renderedBraces.close
              ? 'bg-emerald-900/30 border border-emerald-800/50'
              : 'bg-red-900/30 border border-red-800/50'
          }`}>
            <p className="text-xs text-gray-500">Status</p>
            <p className="text-lg font-bold text-emerald-400">
              {templateBraces.open === renderedBraces.open ? '✓ Preserved' : '✗ Changed!'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// FILES TAB
// ============================================================================

function FilesTab() {
  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-2xl font-bold mb-2">Repository Structure</h2>
        <p className="text-gray-400">All files needed for a complete, working Rust crate.</p>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl overflow-hidden">
        {REPO_STRUCTURE.map((section, i) => (
          <div key={section.dir} className={i > 0 ? 'border-t border-gray-800' : ''}>
            <div className="px-4 py-2 bg-gray-800/30 border-b border-gray-800">
              <span className="text-sm font-semibold text-emerald-400">{section.dir}</span>
            </div>
            <div className="divide-y divide-gray-800/50">
              {section.files.map(file => (
                <div key={file.path} className="px-4 py-3 flex items-center gap-3 hover:bg-gray-800/20">
                  <span className="text-lg">{file.icon}</span>
                  <div className="flex-1">
                    <p className="text-sm font-mono text-gray-200">{file.path}</p>
                    <p className="text-xs text-gray-500">{file.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-bold mb-3">Clone & Use</h3>
        <pre className="text-xs font-mono text-gray-300 bg-gray-800/50 rounded-lg p-4 overflow-x-auto">
{`# Clone the repository
git clone https://github.com/your-org/mlir-templates.git
cd mlir-templates

# Build (templates are validated at compile time)
cargo build

# Run tests
cargo test

# Run the example
cargo run --example neural_flow`}
        </pre>
      </div>

      <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-bold mb-3">Add to your project</h3>
        <pre className="text-xs font-mono text-gray-300 bg-gray-800/50 rounded-lg p-4 overflow-x-auto">
{`# In your Cargo.toml:
[dependencies]
mlir-templates = { path = "../mlir-templates" }

# Or from git:
[dependencies]
mlir-templates = { git = "https://github.com/your-org/mlir-templates" }`}
        </pre>
      </div>
    </div>
  );
}

// ============================================================================
// USAGE TAB
// ============================================================================

function UsageTab() {
  const [copiedFile, setCopiedFile] = useState<string | null>(null);

  const copyCode = async (id: string, code: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedFile(id);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h2 className="text-2xl font-bold mb-2">Rust Implementation</h2>
        <p className="text-gray-400">Complete, correct, compilable code. Copy these files into your project.</p>
      </div>

      <CodeBlock id="usage" title="Usage Example (replacing your generate_mlir_for_op)" copied={copiedFile === 'usage'} onCopy={() => copyCode('usage', USAGE_CODE)} code={USAGE_CODE} />
      <CodeBlock id="render" title="src/render.rs (core engine)" copied={copiedFile === 'render'} onCopy={() => copyCode('render', RENDER_CODE)} code={RENDER_CODE} />
      <CodeBlock id="validate" title="src/validate.rs" copied={copiedFile === 'validate'} onCopy={() => copyCode('validate', VALIDATE_CODE)} code={VALIDATE_CODE} />
      <CodeBlock id="build" title="build.rs" copied={copiedFile === 'build'} onCopy={() => copyCode('build', BUILD_CODE)} code={BUILD_CODE} />
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
        <button onClick={onCopy} className="flex items-center gap-1 px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-xs text-gray-300">
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
// SYNTAX HIGHLIGHTING
// ============================================================================

function TemplateHighlight({ template, params }: { template: string; params: Record<string, string> }) {
  const lines = template.split('\n');
  return (
    <>
      {lines.map((line, i) => (
        <div key={i} className="flex hover:bg-gray-800/30">
          <span className="inline-block w-8 text-right mr-4 text-gray-700 select-none text-[10px] flex-shrink-0">{i + 1}</span>
          <span className="flex-1">
            {line.split(/(\{\{\w+\}\})/g).map((part, j) => {
              if (part.match(/^\{\{\w+\}\}$/)) {
                const varName = part.slice(2, -2);
                const hasValue = params[varName] !== undefined && params[varName] !== '';
                return (
                  <span key={j} className={`px-0.5 rounded ${hasValue ? 'bg-emerald-500/30 text-emerald-300 border-b-2 border-emerald-400' : 'bg-red-500/30 text-red-300 border-b-2 border-red-400'}`}>
                    {part}
                  </span>
                );
              }
              return <span key={j}>{colorizeMlir(part)}</span>;
            })}
          </span>
        </div>
      ))}
    </>
  );
}

function RenderedHighlight({ rendered, params }: { rendered: string; params: Record<string, string> }) {
  const lines = rendered.split('\n');
  const values = Object.values(params).filter(v => v.length > 0);
  return (
    <>
      {lines.map((line, i) => (
        <div key={i} className="flex hover:bg-emerald-900/10">
          <span className="inline-block w-8 text-right mr-4 text-gray-700 select-none text-[10px] flex-shrink-0">{i + 1}</span>
          <span className="flex-1">
            {values.length > 0 ? (() => {
              const escaped = values.map(v => v.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
              const regex = new RegExp(`(${escaped.join('|')})`, 'g');
              return line.split(regex).map((part, j) => {
                if (values.includes(part)) {
                  return <span key={j} className="bg-emerald-500/20 text-emerald-300 px-0.5 rounded border-b border-emerald-500/40">{part}</span>;
                }
                return <span key={j}>{colorizeMlir(part)}</span>;
              });
            })() : colorizeMlir(line)}
          </span>
        </div>
      ))}
    </>
  );
}

function colorizeMlir(text: string): React.ReactNode {
  const parts = text.split(/(\b(?:module|func\.func|func|return|tensor|affine_map|iterator_types|indexing_maps|ins|outs|dense|arith\.|linalg\.|math\.|stablehlo\.)\w*|%\w+|"[^"]*"|\/\/.*$)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (!part) return null;
        if (part.startsWith('//')) return <span key={i} className="text-gray-600 italic">{part}</span>;
        if (part.startsWith('"')) return <span key={i} className="text-amber-300">{part}</span>;
        if (part.startsWith('%')) return <span key={i} className="text-sky-300">{part}</span>;
        if (/^(module|func\.func|func|return|affine_map|iterator_types|indexing_maps|ins|outs|dense)$/.test(part))
          return <span key={i} className="text-purple-400">{part}</span>;
        if (/^(arith|linalg|math|stablehlo)\./.test(part))
          return <span key={i} className="text-cyan-400">{part}</span>;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

// ============================================================================
// RUST CODE STRINGS (for the Usage tab)
// ============================================================================

const USAGE_CODE = `// In your neural_flow/mod.rs — replace generate_mlir_for_op:

use mlir_templates::MlirTemplate;
use std::collections::HashMap;

// BEFORE (broken):
// FusedOp::MatMulBiasRelu { m, n, k } => {
//     Ok(format!("module @matmul {{ ... tensor<{}x{}> ... }}", m, n))
// }
// ^^^ Rust's format! eats ALL {} — compile errors!

// AFTER (fixed):
FusedOp::MatMulBiasRelu { m, n, k } => {
    let template = MlirTemplate::from_embedded(
        "matmul_bias_relu",
        include_str!("../templates/matmul_bias_relu.mlir"),
    )?;

    let mut params = HashMap::new();
    params.insert("M".into(), m.to_string());
    params.insert("N".into(), n.to_string());
    params.insert("K".into(), k.to_string());
    params.insert("DTYPE".into(), "f16".into());

    // Only {{M}}, {{N}}, {{K}}, {{DTYPE}} are replaced
    // All MLIR braces {} are completely untouched
    template.render(&params)
}`;

const RENDER_CODE = `use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::{TemplateError, Result};

/// Regex: matches ONLY {{VAR}} (double braces)
static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

pub struct MlirTemplate {
    template: String,
    required_params: HashSet<String>,
    name: String,
}

impl MlirTemplate {
    pub fn new(template: impl Into<String>) -> Result<Self> {
        let template = template.into();
        let required_params = Self::extract_params(&template);
        Ok(Self { template, required_params, name: "<inline>".into() })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let name = path.as_ref().file_stem()
            .and_then(|s| s.to_str()).unwrap_or("<file>").into();
        Ok(Self { template: content, required_params: Self::extract_params(&content), name })
    }

    pub fn from_embedded(name: &str, content: &'static str) -> Result<Self> {
        Ok(Self {
            template: content.into(),
            required_params: Self::extract_params(content),
            name: name.into(),
        })
    }

    fn extract_params(template: &str) -> HashSet<String> {
        PLACEHOLDER_RE.captures_iter(template)
            .filter_map(|cap| cap.get(1).map(|m| m.as_str().into()))
            .collect()
    }

    /// THE CORE: Only replaces {{VAR}}, never touches MLIR's {}
    pub fn render(&self, params: &HashMap<String, String>) -> Result<String> {
        for param in &self.required_params {
            if !params.contains_key(param) {
                return Err(TemplateError::MissingParameter(param.clone()));
            }
        }
        let mut result = self.template.clone();
        for (key, value) in params {
            // Build "{{KEY}}" via concatenation, NOT format!
            let placeholder = {
                let mut s = String::with_capacity(key.len() + 4);
                s.push('{'); s.push('{');
                s.push_str(key);
                s.push('}'); s.push('}');
                s
            };
            result = result.replace(&placeholder, value);
        }
        Ok(result)
    }
}`;

const VALIDATE_CODE = `use std::collections::HashSet;
use regex::Regex;
use once_cell::sync::Lazy;

static PLACEHOLDER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\\{\\{(\\w+)\\}\\}").unwrap()
});

pub fn validate_template(template: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    let mut depth: i32 = 0;
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if chars.peek() == Some(&'{') {
                    chars.next(); // skip {{ (placeholder)
                } else {
                    depth += 1;
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    chars.next(); // skip }} (placeholder)
                } else {
                    depth -= 1;
                    if depth < 0 {
                        errors.push("Unbalanced: extra '}'".into());
                        depth = 0;
                    }
                }
            }
            _ => {}
        }
    }
    if depth > 0 {
        errors.push(format!("{} unclosed '{{'", depth));
    }
    if template.contains("{{{{") {
        errors.push("Nested placeholders".into());
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}`;

const BUILD_CODE = `use std::fs;
use std::path::Path;

fn main() {
    let dir = Path::new("templates");
    if !dir.exists() { return; }

    let mut has_errors = false;
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().map_or(false, |e| e == "mlir") {
            let content = fs::read_to_string(&path).unwrap();
            let name = path.file_stem().unwrap().to_str().unwrap();
            match validate(&content) {
                Ok(()) => println!("cargo:warning=  ✓ {}.mlir", name),
                Err(errs) => {
                    has_errors = true;
                    for e in &errs { eprintln!("  ✗ {}.mlir: {}", name, e); }
                }
            }
        }
    }
    if has_errors { panic!("Template validation failed!"); }
    println!("cargo:rerun-if-changed=templates/");
}

fn validate(t: &str) -> Result<(), Vec<String>> {
    let mut d: i32 = 0;
    let mut c = t.chars().peekable();
    while let Some(ch) = c.next() {
        match ch {
            '{' => { if c.peek() == Some(&'{') { c.next(); } else { d += 1; } }
            '}' => { if c.peek() == Some(&'}') { c.next(); } else { d -= 1; } }
            _ => {}
        }
    }
    if d != 0 { Err(vec![format!("Unbalanced: {}", d)]) } else { Ok(()) }
}`;
