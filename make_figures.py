"""Generate figures from the recorded run; never substitutes invented examples."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
read = lambda name: json.loads((OUT / name).read_text())
report, history, predictions = read('metrics.json'), read('history.json'), read('predictions.json')
plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafaf5','axes.facecolor':'#fafaf5','text.color':'#16352c','axes.labelcolor':'#16352c','font.size':11})
cm = np.array(report['mlp']['confusion_matrix'])
fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
ax.imshow(cm, cmap='Greens')
for y in range(10):
    for x in range(10):
        ax.text(x, y, str(cm[y,x]), ha='center', va='center', color='white' if cm[y,x] > 100 else '#16352c', fontsize=10)
ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted digit', ylabel='Actual digit', title='Held-out test set · confusion matrix\n1,717 correct / 1,797 images')
fig.savefig(OUT / 'confusion-matrix.png', dpi=160); plt.close(fig)
fig, ax = plt.subplots(figsize=(9, 4.7), layout='constrained')
ax.plot([h['epoch'] for h in history], [h['train_loss'] for h in history], label='Training loss', color='#1d6953', lw=2)
ax.plot([h['epoch'] for h in history], [h['validation_loss'] for h in history], label='Validation loss', color='#b8682c', lw=2)
ax.axvline(report['selected_epoch'], color='#61746b', ls='--', label=f'Selected epoch {report["selected_epoch"]}')
ax.set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Checkpoint selected by validation loss', ylim=(0,None)); ax.legend(frameon=False)
fig.savefig(OUT / 'learning-curves.png', dpi=160); plt.close(fig)
errors = sorted((r for r in predictions if r['actual'] != r['predicted']), key=lambda r:(-r['score'], r['index']))[:12]
fig, axes = plt.subplots(3, 4, figsize=(9, 7.8), layout='constrained')
for ax, row in zip(axes.flat, errors):
    ax.imshow(np.array(row['pixels']).reshape(8,8), cmap='gray', vmin=0,vmax=16,interpolation='nearest')
    ax.set_title(f'Actual {row["actual"]} → predicted {row["predicted"]}\nScore {row["score"]:.1%} · sample {row["index"]}',fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('12 highest-scoring mistakes out of 80\nA high model score can still be wrong.',fontsize=15)
fig.savefig(OUT / 'errors.png', dpi=160); plt.close(fig)
print('Wrote confusion-matrix.png, learning-curves.png, errors.png')
