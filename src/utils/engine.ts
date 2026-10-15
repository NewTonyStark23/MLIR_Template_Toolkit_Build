// MLIR Template Engine - Safe string interpolation for MLIR templates
// Uses {{VAR}} syntax to avoid conflicts with MLIR's curly braces and Rust's format! macro

export interface TemplateParam {
  name: string;
  value: string;
  description: string;
  defaultValue: string;
  type: 'integer' | 'string' | 'dtype' | 'dimension';
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

export interface RenderResult {
  output: string;
  validation: ValidationResult;
  renderTime: number;
  paramCount: number;
}

/**
 * Extract all {{VAR}} placeholders from a template
 */
export function extractParams(template: string): string[] {
  const regex = /\{\{(\w+)\}\}/g;
  const params = new Set<string>();
  let match;
  while ((match = regex.exec(template)) !== null) {
    params.add(match[1]);
  }
  return Array.from(params).sort();
}

/**
 * Render a template with the given parameters
 * Replaces {{VAR}} with the corresponding value
 */
export function renderTemplate(
  template: string,
  params: Record<string, string>
): RenderResult {
  const startTime = performance.now();
  const errors: string[] = [];
  const warnings: string[] = [];

  // Extract required params
  const requiredParams = extractParams(template);

  // Check for missing params
  for (const param of requiredParams) {
    if (!params[param] && params[param] !== '') {
      errors.push(`Missing required parameter: {{${param}}}`);
    }
  }

  // Check for unused params
  for (const key of Object.keys(params)) {
    if (!requiredParams.includes(key) && params[key]) {
      warnings.push(`Unused parameter: ${key}`);
    }
  }

  // Perform substitution
  let output = template;
  for (const [key, value] of Object.entries(params)) {
    const placeholder = `{{${key}}}`;
    output = output.split(placeholder).join(value);
  }

  // Check for unresolved placeholders
  const unresolved = extractParams(output);
  if (unresolved.length > 0) {
    warnings.push(`Unresolved placeholders: ${unresolved.map(p => `{{${p}}}`).join(', ')}`);
  }

  // Validate MLIR structure
  const mlirValidation = validateMlirStructure(output);
  errors.push(...mlirValidation.errors);
  warnings.push(...mlirValidation.warnings);

  const endTime = performance.now();

  return {
    output,
    validation: {
      valid: errors.length === 0,
      errors,
      warnings,
    },
    renderTime: endTime - startTime,
    paramCount: requiredParams.length,
  };
}

/**
 * Validate MLIR structure of rendered output
 */
function validateMlirStructure(output: string): { errors: string[]; warnings: string[] } {
  const errors: string[] = [];
  const warnings: string[] = [];

  // Check balanced braces
  let braceCount = 0;
  for (const char of output) {
    if (char === '{') braceCount++;
    if (char === '}') braceCount--;
    if (braceCount < 0) {
      errors.push('Unbalanced braces: extra closing brace found');
      break;
    }
  }
  if (braceCount > 0) {
    errors.push(`Unbalanced braces: ${braceCount} unclosed opening brace(s)`);
  }

  // Check balanced angle brackets in tensor types
  const tensorMatches = output.match(/tensor<[^>]*>/g);
  if (tensorMatches) {
    for (const match of tensorMatches) {
      if (match.includes('{{')) {
        warnings.push('Tensor type contains unresolved placeholders');
      }
    }
  }

  // Check for common MLIR patterns
  if (output.includes('func.func') || output.includes('func @')) {
    // Has function definition - good
  } else if (output.includes('module')) {
    // Has module but no function - might be okay for some templates
  }

  // Check for valid dimension syntax
  const dimPattern = /tensor<[^>]*>/g;
  let dimMatch;
  while ((dimMatch = dimPattern.exec(output)) !== null) {
    const dimStr = dimMatch[0];
    // Check for invalid dimension values
    if (dimStr.includes('tensor<0') || dimStr.includes('x0x')) {
      warnings.push('Zero dimension detected in tensor type');
    }
  }

  return { errors, warnings };
}

/**
 * Format MLIR output with basic indentation
 */
export function formatMlir(mlir: string): string {
  let indent = 0;
  const lines = mlir.split('\n');
  const formatted: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      formatted.push('');
      continue;
    }

    // Decrease indent for closing braces
    if (trimmed.startsWith('}')) {
      indent = Math.max(0, indent - 1);
    }

    formatted.push('  '.repeat(indent) + trimmed);

    // Increase indent for opening braces
    const openBraces = (trimmed.match(/\{/g) || []).length;
    const closeBraces = (trimmed.match(/\}/g) || []).length;
    indent += openBraces - closeBraces;
    if (trimmed.startsWith('}')) {
      // Already handled above
    }
    indent = Math.max(0, indent);
  }

  return formatted.join('\n');
}
