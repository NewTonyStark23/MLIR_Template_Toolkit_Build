import { useState, useCallback, useMemo } from 'react';
import { TEMPLATES, DEFAULT_TEMPLATE, MlirTemplateDef, TEMPLATE_CATEGORIES } from './data/templates';
import { renderTemplate, extractParams, formatMlir, RenderResult } from './utils/engine';

export default function App() {
  const [selectedTemplate, setSelectedTemplate] = useState<MlirTemplateDef>(DEFAULT_TEMPLATE);
  const [params, setParams] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    DEFAULT_TEMPLATE.params.forEach(p => {
      initial[p.name] = p.defaultValue;
    });
    return initial;
  });
  const [customTemplate, setCustomTemplate] = useState<string>('');
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [showFormatted, setShowFormatted] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [activeCategory, setActiveCategory] = useState<string>('All');
  const [copied, setCopied] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const activeTemplate = useMemo(() => {
    if (isCustomMode) return customTemplate;
    return selectedTemplate.template;
  }, [isCustomMode, customTemplate, selectedTemplate]);

  const renderResult: RenderResult = useMemo(() => {
    return renderTemplate(activeTemplate, params);
  }, [activeTemplate, params]);

  const displayOutput = useMemo(() => {
    if (showFormatted) {
      return formatMlir(renderResult.output);
    }
    return renderResult.output;
  }, [renderResult.output, showFormatted]);

  const templateParams = useMemo(() => {
    if (isCustomMode) {
      const extracted = extractParams(customTemplate);
      return extracted.map(name => ({
        name,
        value: params[name] || '',
        description: `Parameter: ${name}`,
        defaultValue: '',
        type: 'string' as const,
      }));
    }
    return selectedTemplate.params;
  }, [isCustomMode, customTemplate, selectedTemplate.params, params]);

  const filteredTemplates = useMemo(() => {
    let filtered = TEMPLATES;
    if (activeCategory !== 'All') {
      filtered = filtered.filter(t => t.category === activeCategory);
    }
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      filtered = filtered.filter(t =>
        t.name.toLowerCase().includes(q) ||
        t.description.toLowerCase().includes(q) ||
        t.tags.some(tag => tag.toLowerCase().includes(q))
      );
    }
    return filtered;
  }, [activeCategory, searchQuery]);

  const handleSelectTemplate = useCallback((template: MlirTemplateDef) => {
    setSelectedTemplate(template);
    setIsCustomMode(false);
    const newParams: Record<string, string> = {};
    template.params.forEach(p => {
      newParams[p.name] = params[p.name] || p.defaultValue;
    });
    setParams(newParams);
  }, [params]);

  const handleParamChange = useCallback((name: string, value: string) => {
    setParams(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(displayOutput);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = displayOutput;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [displayOutput]);

  const handleDownload = useCallback(() => {
    const blob = new Blob([displayOutput], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedTemplate.id || 'custom'}.mlir`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [displayOutput, selectedTemplate.id]);

  const handleResetParams = useCallback(() => {
    if (!isCustomMode) {
      const newParams: Record<string, string> = {};
      selectedTemplate.params.forEach(p => {
        newParams[p.name] = p.defaultValue;
      });
      setParams(newParams);
    }
  }, [selectedTemplate, isCustomMode]);

  return (
    <div className="h-screen w-screen flex flex-col bg-gray-950 text-gray-100 overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm">
        <div className="flex items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg hover:bg-gray-800 transition-colors lg:hidden"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-400 to-cyan-500 flex items-center justify-center">
                <svg className="w-5 h-5 text-gray-900" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </div>
              <div>
                <h1 className="text-lg font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent">
                  MLIR Template Toolkit
                </h1>
                <p className="text-xs text-gray-500 hidden sm:block">Safe MLIR code generation with {'{{VAR}}'} interpolation</p>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className={`px-3 py-1 rounded-full text-xs font-medium ${
              renderResult.validation.valid 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                : 'bg-red-500/20 text-red-400 border border-red-500/30'
            }`}>
              {renderResult.validation.valid ? '✓ Valid' : '✗ Errors'}
            </div>
            <div className="px-3 py-1 rounded-full text-xs font-medium bg-gray-800 text-gray-400 border border-gray-700">
              {renderResult.renderTime.toFixed(2)}ms
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar - Template Library */}
        <aside className={`${sidebarOpen ? 'w-80' : 'w-0'} flex-shrink-0 border-r border-gray-800 bg-gray-900/50 flex flex-col overflow-hidden transition-all duration-300`}>
          <div className="p-3 border-b border-gray-800">
            <div className="relative">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                placeholder="Search templates..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-9 pr-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-sm text-gray-200 placeholder-gray-500 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20"
              />
            </div>
            {/* Category tabs */}
            <div className="flex gap-1 mt-2 overflow-x-auto pb-1 scrollbar-hide">
              <button
                onClick={() => setActiveCategory('All')}
                className={`px-2 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
                  activeCategory === 'All' ? 'bg-emerald-500/20 text-emerald-400' : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                All
              </button>
              {TEMPLATE_CATEGORIES.map(cat => (
                <button
                  key={cat}
                  onClick={() => setActiveCategory(cat)}
                  className={`px-2 py-1 rounded text-xs font-medium whitespace-nowrap transition-colors ${
                    activeCategory === cat ? 'bg-emerald-500/20 text-emerald-400' : 'text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Template list */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {filteredTemplates.map(template => (
              <button
                key={template.id}
                onClick={() => handleSelectTemplate(template)}
                className={`w-full text-left p-3 rounded-lg transition-all ${
                  !isCustomMode && selectedTemplate.id === template.id
                    ? 'bg-emerald-500/10 border border-emerald-500/30 shadow-lg shadow-emerald-500/5'
                    : 'hover:bg-gray-800/50 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-gray-200">{template.name}</span>
                </div>
                <p className="text-xs text-gray-500 mt-1 line-clamp-2">{template.description}</p>
                <div className="flex gap-1 mt-2 flex-wrap">
                  {template.tags.slice(0, 3).map(tag => (
                    <span key={tag} className="px-1.5 py-0.5 bg-gray-800 rounded text-[10px] text-gray-400">
                      {tag}
                    </span>
                  ))}
                </div>
              </button>
            ))}
            {/* Custom template button */}
            <button
              onClick={() => {
                setIsCustomMode(true);
                setCustomTemplate(`module @custom {\n  func.func @main(%arg0: tensor<{{M}}x{{N}}x{{DTYPE}}>) -> tensor<{{M}}x{{N}}x{{DTYPE}}> {\n    // Your custom MLIR here\n    return %arg0 : tensor<{{M}}x{{N}}x{{DTYPE}}>\n  }\n}`);
              }}
              className={`w-full text-left p-3 rounded-lg transition-all border border-dashed ${
                isCustomMode
                  ? 'bg-cyan-500/10 border-cyan-500/30'
                  : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/30'
              }`}
            >
              <div className="flex items-center gap-2">
                <svg className="w-4 h-4 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <span className="text-sm font-medium text-cyan-400">Custom Template</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">Write your own MLIR template</p>
            </button>
          </div>
        </aside>

        {/* Main workspace */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Toolbar */}
          <div className="flex-shrink-0 flex items-center justify-between px-4 py-2 border-b border-gray-800 bg-gray-900/30">
            <div className="flex items-center gap-2">
              {!isCustomMode && (
                <h2 className="text-sm font-semibold text-gray-300">{selectedTemplate.name}</h2>
              )}
              {isCustomMode && (
                <h2 className="text-sm font-semibold text-cyan-400">Custom Template</h2>
              )}
            </div>
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1.5 text-xs text-gray-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showFormatted}
                  onChange={(e) => setShowFormatted(e.target.checked)}
                  className="rounded border-gray-600 bg-gray-800 text-emerald-500 focus:ring-emerald-500/20"
                />
                Format output
              </label>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-xs font-medium text-gray-300 transition-colors"
              >
                {copied ? (
                  <>
                    <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Copied!
                  </>
                ) : (
                  <>
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    Copy
                  </>
                )}
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-xs font-medium text-gray-300 transition-colors"
              >
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download
              </button>
            </div>
          </div>

          {/* Panels */}
          <div className="flex-1 flex overflow-hidden">
            {/* Left panel - Template + Parameters */}
            <div className="w-1/2 flex flex-col border-r border-gray-800 overflow-hidden">
              {/* Template editor */}
              <div className="flex-1 flex flex-col overflow-hidden">
                <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800 bg-gray-900/30">
                  <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                    {isCustomMode ? 'Template Editor' : 'Template Source'}
                  </span>
                  <span className="text-xs text-gray-600">
                    {templateParams.length} params
                  </span>
                </div>
                <div className="flex-1 overflow-auto">
                  {isCustomMode ? (
                    <textarea
                      value={customTemplate}
                      onChange={(e) => setCustomTemplate(e.target.value)}
                      className="w-full h-full min-h-[300px] p-4 bg-transparent text-sm font-mono text-gray-300 resize-none focus:outline-none leading-relaxed"
                      spellCheck={false}
                    />
                  ) : (
                    <pre className="p-4 text-sm font-mono text-gray-300 leading-relaxed whitespace-pre-wrap overflow-auto h-full">
                      <HighlightedTemplate template={activeTemplate} params={params} />
                    </pre>
                  )}
                </div>
              </div>

              {/* Parameters panel */}
              <div className="flex-shrink-0 border-t border-gray-800 max-h-[40%] overflow-auto">
                <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800 bg-gray-900/30">
                  <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">Parameters</span>
                  <button
                    onClick={handleResetParams}
                    className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
                  >
                    Reset defaults
                  </button>
                </div>
                <div className="p-3 grid grid-cols-2 gap-2">
                  {templateParams.map(param => (
                    <div key={param.name} className="group">
                      <label className="flex items-center gap-1.5 mb-1">
                        <code className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-1 rounded">
                          {`{{${param.name}}}`}
                        </code>
                        <span className="text-[10px] text-gray-600 truncate" title={param.description}>
                          {param.description}
                        </span>
                      </label>
                      <input
                        type="text"
                        value={params[param.name] || ''}
                        onChange={(e) => handleParamChange(param.name, e.target.value)}
                        placeholder={param.defaultValue}
                        className="w-full px-2 py-1.5 bg-gray-800/50 border border-gray-700 rounded text-xs font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/20 transition-colors"
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right panel - Output */}
            <div className="w-1/2 flex flex-col overflow-hidden">
              <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800 bg-gray-900/30">
                <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                  Rendered Output
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-gray-600">
                    {displayOutput.split('\n').length} lines
                  </span>
                  <span className="text-[10px] text-gray-600">
                    {(displayOutput.length / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
              <div className="flex-1 overflow-auto">
                <pre className="p-4 text-sm font-mono text-gray-300 leading-relaxed whitespace-pre-wrap">
                  <OutputHighlight output={displayOutput} />
                </pre>
              </div>

              {/* Validation panel */}
              {(renderResult.validation.errors.length > 0 || renderResult.validation.warnings.length > 0) && (
                <div className="flex-shrink-0 border-t border-gray-800 max-h-[30%] overflow-auto">
                  <div className="px-3 py-2 border-b border-gray-800 bg-gray-900/30">
                    <span className="text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Validation
                    </span>
                  </div>
                  <div className="p-3 space-y-1">
                    {renderResult.validation.errors.map((err, i) => (
                      <div key={`err-${i}`} className="flex items-start gap-2 text-xs">
                        <span className="text-red-400 mt-0.5">●</span>
                        <span className="text-red-300">{err}</span>
                      </div>
                    ))}
                    {renderResult.validation.warnings.map((warn, i) => (
                      <div key={`warn-${i}`} className="flex items-start gap-2 text-xs">
                        <span className="text-amber-400 mt-0.5">●</span>
                        <span className="text-amber-300">{warn}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="flex-shrink-0 border-t border-gray-800 bg-gray-900/50 px-4 py-1.5">
        <div className="flex items-center justify-between text-[10px] text-gray-600">
          <span>MLIR Template Toolkit v1.0 — Solves format! macro curly brace conflicts</span>
          <span>Uses {'{{VAR}}'} syntax for safe interpolation</span>
        </div>
      </footer>
    </div>
  );
}

// Syntax highlighting component for template
function HighlightedTemplate({ template, params }: { template: string; params: Record<string, string> }) {
  const parts = template.split(/(\{\{\w+\}\})/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.match(/^\{\{\w+\}\}$/)) {
          const paramName = part.slice(2, -2);
          const hasValue = params[paramName] && params[paramName].length > 0;
          return (
            <span
              key={i}
              className={`px-0.5 rounded ${
                hasValue
                  ? 'bg-emerald-500/20 text-emerald-300 border-b border-emerald-500/40'
                  : 'bg-red-500/20 text-red-300 border-b border-red-500/40'
              }`}
              title={hasValue ? `= ${params[paramName]}` : 'Missing value'}
            >
              {part}
            </span>
          );
        }
        return <span key={i}>{highlightMlirSyntax(part)}</span>;
      })}
    </>
  );
}

// Syntax highlighting for output
function OutputHighlight({ output }: { output: string }) {
  const lines = output.split('\n');
  return (
    <>
      {lines.map((line, i) => (
        <div key={i} className="flex">
          <span className="inline-block w-8 text-right mr-4 text-gray-700 select-none text-xs leading-relaxed">
            {i + 1}
          </span>
          <span>{highlightMlirSyntax(line)}</span>
        </div>
      ))}
    </>
  );
}

// Basic MLIR syntax highlighting
function highlightMlirSyntax(text: string): React.ReactNode {
  // Split by keywords and patterns
  const keywords = /\b(module|func\.func|func|return|tensor|affine_map|iterator_types|indexing_maps|ins|outs|dense|arith\.|linalg\.|math\.|stablehlo\.|scf\.|memref\.|vector\.|cf\.|call|constant|fill|matmul|batch_matmul|generic|yield|custom_call)\b/g;
  const types = /\b(tensor<[^>]+>|f16|f32|f64|bf16|i1|i8|i16|i32|i64|index)\b/g;
  const strings = /"[^"]*"/g;
  const numbers = /\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;
  const comments = /\/\/.*$/gm;
  const values = /%\w+/g;
  const attributes = /#\w+[\w.<>,\s]*/g;

  // Simple approach: colorize in order of priority
  let result = text;
  const replacements: { start: number; end: number; className: string; text: string }[] = [];

  // Find comments
  let match;
  const commentRegex = /\/\/.*/g;
  while ((match = commentRegex.exec(text)) !== null) {
    replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-gray-600 italic', text: match[0] });
  }

  // Find strings
  const stringRegex = /"[^"]*"/g;
  while ((match = stringRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-amber-300', text: match[0] });
    }
  }

  // Find SSA values
  const valueRegex = /%\w+/g;
  while ((match = valueRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-sky-300', text: match[0] });
    }
  }

  // Find keywords
  const keywordRegex = /\b(module|func\.func|func|return|affine_map|iterator_types|indexing_maps|ins|outs|dense|call_target_name)\b/g;
  while ((match = keywordRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-purple-400 font-medium', text: match[0] });
    }
  }

  // Find dialect operations
  const dialectRegex = /\b(arith|linalg|math|stablehlo|scf|memref|vector|cf)\.\w+/g;
  while ((match = dialectRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-cyan-400', text: match[0] });
    }
  }

  // Find types
  const typeRegex = /\b(f16|f32|f64|bf16|i1|i8|i16|i32|i64|index)\b/g;
  while ((match = typeRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-emerald-400', text: match[0] });
    }
  }

  // Find numbers
  const numberRegex = /\b(\d+\.?\d*(?:e[+-]?\d+)?)\b/g;
  while ((match = numberRegex.exec(text)) !== null) {
    if (!isOverlapping(replacements, match.index, match.index + match[0].length)) {
      replacements.push({ start: match.index, end: match.index + match[0].length, className: 'text-orange-300', text: match[0] });
    }
  }

  // Sort by start position
  replacements.sort((a, b) => a.start - b.start);

  // Build result
  if (replacements.length === 0) {
    return <span>{text}</span>;
  }

  const elements: React.ReactNode[] = [];
  let pos = 0;
  for (const rep of replacements) {
    if (rep.start > pos) {
      elements.push(<span key={`t-${pos}`}>{text.slice(pos, rep.start)}</span>);
    }
    elements.push(<span key={`r-${rep.start}`} className={rep.className}>{rep.text}</span>);
    pos = rep.end;
  }
  if (pos < text.length) {
    elements.push(<span key={`t-${pos}`}>{text.slice(pos)}</span>);
  }

  return <>{elements}</>;
}

function isOverlapping(replacements: { start: number; end: number }[], start: number, end: number): boolean {
  return replacements.some(r => start < r.end && end > r.start);
}
