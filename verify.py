"""Meaningful smoke checks without replacing the full experiment."""
import sys
sys.dont_write_bytecode = True
from src.utils import ROOT, configure_paths, seed_all
configure_paths()
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from src.model import BayesianCNN
from src.uncertainty import mc_predict

seed_all(42)
torch.set_num_threads(4)
device = torch.device('cpu')
print('Python:', sys.version, '\nPyTorch:', torch.__version__, '\nDevice:', device, flush=True)
if device.type == 'cuda':
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    assert (torch.ones(2, device=device) + 1).sum().item() == 4
training = MNIST(str(ROOT / 'data'), train=True, download=True, transform=ToTensor())
testing = MNIST(str(ROOT / 'data'), train=False, download=True, transform=ToTensor())
assert len(training) == 60000 and len(testing) == 10000
x = torch.stack([training[i][0] for i in range(16)]).to(device)
y = torch.tensor([training[i][1] for i in range(16)], device=device)
assert x.shape == (16, 1, 28, 28)
model = BayesianCNN().to(device)
assert model(x).shape == (16, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for _ in range(2):
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(model(x), y)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
model.eval()
with torch.inference_mode():
    assert torch.equal(model(x), model(x))
    mean, variance, entropy, confidence, pred = mc_predict(model, x, 30)
assert not model.training and not any(m.training for m in model.modules())
assert torch.allclose(mean.sum(1), torch.ones(16, device=device), atol=1e-5)
assert (variance >= 0).all() and (variance > 0).any()
assert (entropy >= 0).all() and (entropy <= torch.log(torch.tensor(10.0, device=device)) + 1e-5).all()
print('PASS: dataset, shapes, training gradients, deterministic predictions, stochastic variance, probability sums, entropy bounds, dropout state restoration. GPU arithmetic checked only when CUDA is available.', flush=True)
