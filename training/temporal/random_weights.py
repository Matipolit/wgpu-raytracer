"""Random (non-trivial) weights for parity testing: random_weights.py OUT.pt"""
import sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as M  # noqa: E402
torch.manual_seed(3)
net = M.Net()
torch.nn.init.normal_(net.head.weight, std=0.3)
torch.nn.init.normal_(net.head.bias, std=0.5)
torch.save(net.state_dict(), sys.argv[1])
