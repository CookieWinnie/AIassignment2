"""Render report formulas with existing Matplotlib mathtext, without installing fonts."""
import sys
sys.dont_write_bytecode = True
from src.utils import ROOT, configure_paths
configure_paths()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

FORMULAS = [
    r'P(w\mid D)=\frac{P(D\mid w)P(w)}{P(D)}',
    r'P(y\mid x,D)=\int P(y\mid x,w)P(w\mid D)\,dw',
    r'\bar{p}=\frac{1}{T}\sum_{t=1}^{T}p_t',
    r'\frac{1}{T}\sum_{t=1}^{T}(p_t-\bar{p})^2',
    r'H(\bar{p})=-\sum_k\bar{p}_k\log(\bar{p}_k)',
    r'\mathrm{argmax}(\bar{p})',
    r'\max(\bar{p})',
]
directory = ROOT / 'report_material/equations'
directory.mkdir(exist_ok=True)
for i, formula in enumerate(FORMULAS):
    fig = plt.figure(figsize=(8, 1), dpi=300)
    text = fig.text(0.01, 0.5, '$'+formula+'$', fontsize=11, color='black')
    fig.canvas.draw()
    bounds = text.get_window_extent(renderer=fig.canvas.get_renderer())
    bbox = Bbox.from_bounds(bounds.x0/300, bounds.y0/300, bounds.width/300, bounds.height/300)
    fig.savefig(directory / f'equation_{i}.png', dpi=300, bbox_inches=bbox, pad_inches=0.02, transparent=True)
    plt.close(fig)
print('Seven mathematical formulas rendered at 300 dpi.')
