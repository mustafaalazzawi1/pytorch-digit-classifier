import { validateModel, predict } from './inference.mjs';
const $ = id => document.getElementById(id);
const base = new URL(document.body.dataset.assets, document.baseURI);
let model, rows, visible, position = 0;
function render() {
  const row = visible[position], scores = predict(model, row.pixels);
  const ranking = scores.map((score, digit) => ({score, digit})).sort((a,b) => b.score - a.score);
  const correct = ranking[0].digit === row.actual;
  const context = $('digit').getContext('2d');
  row.pixels.forEach((pixel, i) => {const v = Math.round(pixel / 16 * 255); context.fillStyle = `rgb(${v},${v},${v})`; context.fillRect(i % 8 * 32, Math.floor(i / 8) * 32, 32, 32);});
  $('digit').setAttribute('aria-label', `Test image ${row.index}: actual digit ${row.actual}, predicted ${ranking[0].digit}`);
  $('sample-label').textContent = `Test sample #${row.index} · original 8 × 8 pixels`;
  $('position').textContent = `${position + 1} of ${visible.length.toLocaleString()} images`;
  $('outcome').textContent = correct ? 'CORRECT PREDICTION' : 'MISCLASSIFIED IMAGE';
  $('outcome').classList.toggle('error-text', !correct);
  $('predicted').textContent = ranking[0].digit;
  $('actual').textContent = row.actual;
  $('scores').replaceChildren(...ranking.slice(0,3).map(({score,digit}) => {
    const li=document.createElement('li'), label=document.createElement('span'), track=document.createElement('span'), bar=document.createElement('span'), value=document.createElement('strong');
    label.textContent=digit;track.className='score-track';track.setAttribute('aria-hidden','true');bar.style.width=`${score*100}%`;value.textContent=`${(score*100).toFixed(1)}%`;track.append(bar);li.append(label,track,value);return li;
  }));
  $('baseline').textContent = `Linear baseline predicted ${row.baseline_predicted} — ${row.baseline_predicted === row.actual ? 'correct' : 'incorrect'} on this image.`;
  $('previous').disabled = position === 0;
  $('next').disabled = position === visible.length - 1;
}
$('filter').addEventListener('change', () => {
  visible = rows.filter(r => $('filter').value === 'all' || ($('filter').value === 'errors' ? r.actual !== r.predicted : r.actual === r.predicted));
  position=0;render();
});
$('previous').addEventListener('click',()=>{if(position>0){position--;render();}});
$('next').addEventListener('click',()=>{if(position<visible.length-1){position++;render();}});
try {
  const read = async name => {const response=await fetch(new URL(name,base));if(!response.ok)throw new Error('Download failed');return response.json();};
  [model,rows] = await Promise.all([read('model.json'),read('predictions.json')]);
  validateModel(model);
  if(!Array.isArray(rows)||rows.length!==1797)throw new Error('Invalid examples');
  visible=rows;render();$('filter').disabled=false;
  $('load-status').textContent='Model ready. Predictions are computed in your browser. Examples remain in original test-set order.';
} catch(error) {
  $('position').textContent='Explorer unavailable';
  $('load-status').textContent='The model or example files could not load. Reload to try again, or use the figures and source links below.';
  $('load-status').classList.add('error-text');
  $('baseline').textContent='The recorded figures below remain available.';
}
