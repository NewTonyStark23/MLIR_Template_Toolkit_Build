import JSZip from 'jszip';
import { saveAs } from 'file-saver';

// Import all files from the mlir-templates directory as raw strings
// Vite's import.meta.glob with ?raw loads file contents as strings at build time
const rawFiles = import.meta.glob(
  '/mlir-templates/**/*',
  { eager: true, query: '?raw', import: 'default' }
) as Record<string, string>;

/**
 * Build a zip file containing the entire mlir-templates Rust crate
 * and trigger a download in the browser.
 */
export async function downloadCrateZip(): Promise<void> {
  const zip = new JSZip();

  // Convert the Vite glob paths to proper zip paths
  // Vite returns paths like "/mlir-templates/src/lib.rs"
  // We want them as "mlir-templates/src/lib.rs" in the zip
  const entries = Object.entries(rawFiles);

  if (entries.length === 0) {
    throw new Error('No files found to package');
  }

  for (const [path, content] of entries) {
    // Strip leading slash
    const zipPath = path.startsWith('/') ? path.slice(1) : path;
    zip.file(zipPath, content);
  }

  // Generate the zip blob
  const blob = await zip.generateAsync({
    type: 'blob',
    compression: 'DEFLATE',
    compressionOptions: { level: 9 },
  });

  // Trigger download
  saveAs(blob, 'mlir-templates.zip');
}

/**
 * Get the list of files that will be included in the download
 */
export function getIncludedFiles(): string[] {
  return Object.keys(rawFiles)
    .map(p => (p.startsWith('/') ? p.slice(1) : p))
    .sort();
}

/**
 * Get the total size of all files (uncompressed)
 */
export function getTotalSize(): number {
  return Object.values(rawFiles).reduce((sum, content) => sum + content.length, 0);
}
