import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { validateModel, predict } from '../web/inference.mjs';
const read = name => JSON.parse(readFileSync(new URL(`../artifacts/${name}`, import.meta.url)));
const model = validateModel(read('model.json'));
const rows = read('predictions.json');
let maxError = 0;
for (const row of rows) {
  const scores = predict(model, row.pixels);
  assert.equal(scores.indexOf(Math.max(...scores)), row.predicted);
  assert.ok(Math.abs(scores.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  maxError = Math.max(maxError, Math.abs(Math.max(...scores) - row.score));
}
assert.ok(maxError < 1e-5, `Score discrepancy ${maxError}`);
assert.throws(() => predict(model, [0]));
assert.throws(() => predict(model, Array(64).fill(NaN)));
assert.throws(() => validateModel({...model, pixel_max: 255}));
console.log(`PASS browser inference matches PyTorch on all ${rows.length} test images; max score difference ${maxError}`);
