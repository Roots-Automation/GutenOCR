// Batch KaTeX validator.  Reads JSON-lines from stdin; writes JSON-lines to stdout.
// Input:  {"idx": "42", "formula": "x^2+y^2"}
// Output: {"idx": "42", "ok": true}
//      or {"idx": "42", "ok": false, "error": "..."}
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
    try {
        katex.renderToString(formula, { throwOnError: true, displayMode: false });
        process.stdout.write(JSON.stringify({ idx, ok: true }) + '\n');
    } catch (e) {
        process.stdout.write(JSON.stringify({ idx, ok: false, error: e.message }) + '\n');
    }
});
