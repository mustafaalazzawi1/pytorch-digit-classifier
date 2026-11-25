// Same exported 64 → ReLU(64) → 10 network; inference stays in the browser.
export function validateModel(model) {
  if (model?.format_version !== 1 || model.architecture !== '64-ReLU64-10' || model.pixel_max !== 16) throw new Error('Unsupported model');
  const s = model.state;
  const vector = (a, n) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);
  const matrix = (a, n, m) => Array.isArray(a) && a.length === n && a.every(row => vector(row, m));
  if (!s || !matrix(s['layers.0.weight'], 64, 64) || !vector(s['layers.0.bias'], 64) || !matrix(s['layers.2.weight'], 10, 64) || !vector(s['layers.2.bias'], 10)) throw new Error('Invalid model tensors');
  return model;
}
export function predict(model, pixels) {
  if (!Array.isArray(pixels) || pixels.length !== 64 || !pixels.every(v => Number.isFinite(v) && v >= 0 && v <= 16)) throw new Error('Expected 64 finite pixels in the range 0–16');
  const s = model.state, x = pixels.map(v => v / 16);
  const layer = (weights, bias, values) => weights.map((row, i) => row.reduce((sum, w, j) => sum + w * values[j], bias[i]));
  const hidden = layer(s['layers.0.weight'], s['layers.0.bias'], x).map(v => Math.max(0, v));
  const logits = layer(s['layers.2.weight'], s['layers.2.bias'], hidden);
  const max = Math.max(...logits), exp = logits.map(v => Math.exp(v - max)), total = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / total);
}
