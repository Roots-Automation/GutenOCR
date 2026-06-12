// Batch KaTeX validator.  Reads JSON-lines from stdin; writes JSON-lines to stdout.
// Input:  {"idx": "42", "formula": "x^2+y^2"}
// Output: {"idx": "42", "ok": true}
//      or {"idx": "42", "ok": false, "error": "..."}
//
// Formulas may arrive pre-wrapped with display-math delimiters (\[...\],
// \begin{equation}..., $...$).  This script strips those delimiters and sets
// displayMode accordingly before handing off to KaTeX.
//
// Invoked as: node _katex_runner.js
// Requires:   katex npm package resolvable from this file's directory or any ancestor.
'use strict';

let katex;
try {
    katex = require('katex');
} catch (e) {
    process.stderr.write('katex npm package not found. Install it with: npm install katex\n');
    process.exit(1);
}

const readline = require('readline');

/**
 * Strip outer display-math delimiters and infer displayMode.
 * Also strips \label{...} which is pure bookkeeping and not understood by KaTeX.
 * Returns { math: string, displayMode: boolean }.
 */
function prepareFormula(formula) {
    formula = formula.trim();
    // Strip \label{...} — bookkeeping only, no mathematical content
    formula = formula.replace(/\\label\{[^}]*\}/g, '');

    // \begin{align*}...\end{align*} and similar — display mode, keep as-is.
    // Note: multline/multline* are valid LaTeX but not supported by KaTeX; passing
    // displayMode: true gives KaTeX the best chance to emit a meaningful error vs a
    // mode-related parse failure.
    if (/^\\begin\{(align|gather|multline|flalign|alignat|split)[*]?\}/.test(formula)) {
        return { math: formula, displayMode: true };
    }

    // \[...\]
    if (formula.startsWith('\\[') && formula.endsWith('\\]')) {
        return { math: formula.slice(2, -2).trim(), displayMode: true };
    }

    // \begin{equation}...\end{equation}  (with optional \tag{...})
    const eqMatch = formula.match(/^\\begin\{equation\}([\s\S]*?)(?:\\tag\{[^}]*\})?\\end\{equation\}$/);
    if (eqMatch) {
        return { math: eqMatch[1].trim(), displayMode: true };
    }

    // $...$ inline
    if (formula.startsWith('$') && formula.endsWith('$') && formula.length > 1) {
        return { math: formula.slice(1, -1).trim(), displayMode: false };
    }

    // Bare formula — use display mode so \begin{pmatrix} etc. work
    return { math: formula, displayMode: true };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

rl.on('line', (line) => {
    if (!line.trim()) return;
    let parsed;
    try {
        parsed = JSON.parse(line);
    } catch (e) {
        process.stdout.write(JSON.stringify({ idx: null, ok: false, error: 'invalid JSON input' }) + '\n');
        return;
    }
    const { idx, formula } = parsed;
    const { math, displayMode } = prepareFormula(formula);
    try {
        katex.renderToString(math, { throwOnError: true, displayMode });
        process.stdout.write(JSON.stringify({ idx, ok: true }) + '\n');
    } catch (e) {
        process.stdout.write(JSON.stringify({ idx, ok: false, error: e.message }) + '\n');
    }
});
