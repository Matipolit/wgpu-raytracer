"""Random (non-trivial) weights for parity testing: random_weights.py OUT.pt

A config.json next to OUT.pt selects the architecture (model.py `configure`)."""
import json
import pathlib
import sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as M  # noqa: E402
config = pathlib.Path(sys.argv[1]).with_name("config.json")
if config.exists():
    M.configure(**json.loads(config.read_text()))
torch.manual_seed(3)
net = M.Net()
torch.nn.init.normal_(net.head.weight, std=0.3)
torch.nn.init.normal_(net.head.bias, std=0.5)
if M.EDGE:
    torch.nn.init.normal_(net.edge, mean=1.0, std=1.0)
torch.save(net.state_dict(), sys.argv[1])
