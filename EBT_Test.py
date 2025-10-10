import math, sys
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, AutoConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Utilities
def exists(x): return x is not None

def to_rgb_and_resize(img_size=224):
    # MNIST/FashionMNIST are 1x28x28; ViTs expect 3xHxW with ImageNet normalization
    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])


def get_loaders(dataset: str, img_size: int, batch: int, workers: int) -> Tuple[DataLoader, DataLoader, int]:
    tfm = to_rgb_and_resize(img_size)
    if dataset.lower() == "mnist":
        train = datasets.MNIST("./data", train=True, download=True, transform=tfm)
        test = datasets.MNIST("./data", train=False, download=True, transform=tfm)
        num_classes = 10
    elif dataset.lower() in ["fashion", "fashionmnist", "fmnist"]:
        train = datasets.FashionMNIST("./data", train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST("./data", train=False, download=True, transform=tfm)
        num_classes = 10
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion'")
    train_loader = DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader, num_classes

class FrozenViTBackbone(nn.Module):
    """
    Frozen vision backbone that returns a single embedding per image.
    """
    def __init__(self, img_size: int = 224, hf_model_id: str = "facebook/dinov2-base", hf_pool: str = "cls"):
        super().__init__()
        self.img_size = img_size
        self.source = None
        self.embed_dim = None
        self.hf_pool = hf_pool
        try:
            self.vit = AutoModel.from_pretrained(hf_model_id, trust_remote_code=True)
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad_(False)

            # embedding dim
            if hasattr(self.vit, "config") and hasattr(self.vit.config, "hidden_size"):
                self.embed_dim = int(self.vit.config.hidden_size)
            else:
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, self.img_size, self.img_size)
                    out = self.vit(pixel_values=dummy)
                    if hasattr(out, "last_hidden_state"):
                        self.embed_dim = out.last_hidden_state.shape[-1]
                    elif hasattr(out, "pooler_output"):
                        self.embed_dim = out.pooler_output.shape[-1]
                    else:
                        raise RuntimeError("Cannot infer embed_dim from HF model outputs.")

            self.source = f"hf:{hf_model_id}"
        except Exception as e:
            raise RuntimeError(f"Failed to load HF model '{hf_model_id}': {e}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, E) embedding. For HF models:
          - pooler_output if available
          - else CLS token from last_hidden_state
          - or mean-pool tokens if hf-pool=mean
        """
        with torch.no_grad():
            out = self.vit(pixel_values=x)  # expect (B,3,H,W)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output  # (B, E)
            elif hasattr(out, "last_hidden_state"):
                tokens = out.last_hidden_state  # (B, N, E)
                if self.hf_pool == "mean":
                    feats = tokens.mean(dim=1)
                else:
                    feats = tokens[:, 0]       # CLS
            else:
                raise RuntimeError("HF model outputs lack last_hidden_state/pooler_output.")
            return feats

# EBT: Energy-based head with latent z and iterative inference
class EBTHead(nn.Module):
    """
    EBT module that:
      - Takes frozen backbone representation r_x
      - Initializes latent z0 = g(r_x)
      - Iteratively refines z by gradient descent on E(r_x, z)
      - Produces logits = h(z_T)
    """
    def __init__(self, embed_dim: int, num_classes: int, z_hidden: int = 0):
        super().__init__()
        E = embed_dim
        # initial prediction head g: r_x -> z0
        if z_hidden > 0:
            self.init_pred = nn.Sequential(
                nn.LayerNorm(E),
                nn.Linear(E, z_hidden),
                nn.GELU(),
                nn.Linear(z_hidden, E)
            )
        else:
            self.init_pred = nn.Linear(E, E)

        # energy network f([r_x, z]) -> scalar
        self.energy_net = nn.Sequential(
            nn.LayerNorm(E * 2),
            nn.Linear(E * 2, E),
            nn.GELU(),
            nn.Linear(E, 1)
        )

        # classifier from refined z -> logits
        self.to_logits = nn.Sequential(
            nn.LayerNorm(E),
            nn.Linear(E, num_classes)
        )

    def energy(self, r_x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        xz = torch.cat([r_x, z], dim=-1)   # (B, 2E)
        return self.energy_net(xz).squeeze(-1)  # (B,)

    def refine(self, r_x, z0, steps, lr):
        with torch.enable_grad():                    # allow grads on z even in eval
            z = z0
            for _ in range(max(steps, 0)):
                z = z.detach().requires_grad_(True)
                e = self.energy(r_x, z).sum()
                grad = torch.autograd.grad(e, z, create_graph=self.training)[0]  #grad of e wrt z
                z = z - lr * grad
        return z


    def forward(self, r_x: torch.Tensor, refine_steps: int = 0, refine_lr: float = 1e-2):
        z0 = self.init_pred(r_x)                       # (B, E)
        zT = self.refine(r_x, z0, refine_steps, refine_lr)
        logits = self.to_logits(zT)                    # (B, C)
        e_final = self.energy(r_x, zT)                 # (B,)
        return logits, e_final

class EBTClassifier(nn.Module):
    def __init__(self, backbone: FrozenViTBackbone, num_classes: int, z_hidden: int = 0):
        super().__init__()
        self.backbone = backbone
        self.ebt = EBTHead(embed_dim=backbone.embed_dim, num_classes=num_classes, z_hidden=z_hidden)

    def forward(self, x, refine_steps: int = 0, refine_lr: float = 1e-2):
        r_x = self.backbone(x)
        logits, e = self.ebt(r_x, refine_steps, refine_lr)
        return logits, e

def ebt_hinge_margin_loss(logits, targets, margin: float = 1.0, reduction: str = "mean"):
    """
    Multi-class hinge loss on energies, using energies = -logits.
    For each sample, enforces: E_pos + margin <= E_neg for all negatives.

    Args:
        logits: (B, C) scores from your head (higher = better).
        targets: (B,) long tensor of true class indices.
        margin: scalar margin.
        reduction: "mean" | "sum" | "none".
    Returns:
        scalar loss (or per-sample if reduction="none").
    """
    # energies = -logits
    E = -logits                               # (B, C)
    B = E.size(0)
    E_pos = E[torch.arange(B), targets]       # (B,)

    # hinge over all negatives: max(0, margin + E_pos - E_neg)
    # broadcast E_pos over classes
    raw = margin + E_pos.unsqueeze(1) - E     # (B, C)
    # do not penalize the positive class itself
    raw[torch.arange(B), targets] = 0.0
    # ReLU to keep only violations
    per_class = F.relu(raw)                   # (B, C)

    # reduce across classes (sum of violations) then across batch
    per_sample = per_class.sum(dim=1)         # (B,)
    #if prefer hard-negative mining instead of summing all violations, swap the sum(dim=1) with max(dim=1).values

    if reduction == "mean":
        return per_sample.mean()
    elif reduction == "sum":
        return per_sample.sum()
    else:
        return per_sample  # "none"

def train_ebt(model: EBTClassifier, loader: DataLoader, opt, device, refine_steps: int, refine_lr: float, e_weight=0.1):
    model.train()
    total_loss = 0.0
    total_ok = 0
    total = 0
    b=0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits, e = model(x, refine_steps=refine_steps, refine_lr=refine_lr)
        loss = ebt_hinge_margin_loss(logits, y, margin=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        total_loss += loss.item() * x.size(0)
        total_ok += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
        if b%50==0:
          print(f"batch: {b} : loss : {loss.item()} ")
        b+=1
    return total_loss/total, total_ok/total

def eval_ebt(model: EBTClassifier, loader: DataLoader, device, refine_steps: int, refine_lr: float, e_weight=0.1):
    model.eval()
    total_loss = 0.0
    total_ok = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, e = model(x, refine_steps=refine_steps, refine_lr=refine_lr)
            loss = ebt_hinge_margin_loss(logits, y, margin=1.0)
            total_loss += loss.item() * x.size(0)
            total_ok += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss/total, total_ok/total

# Data
img_size=224
train_loader, test_loader, num_classes = get_loaders("fashion",img_size, 256,4)

# Backbone
backbone = FrozenViTBackbone(
    img_size=img_size,
    hf_model_id="facebook/dinov2-base",
    hf_pool="cls"
)
print(f"[INFO] Loaded backbone: {backbone.source} | embed_dim={backbone.embed_dim}")

# EBT
ebt = EBTClassifier(backbone, num_classes, z_hidden=128).to(device)
opt_ebt = torch.optim.AdamW(ebt.parameters(), lr=1e-3, weight_decay=0.05)

print(">>>> Training EBT ")
for ep in range(1, 6):
    tr_loss, tr_acc = train_ebt(
        ebt, train_loader, opt_ebt, device,
        refine_steps=3, refine_lr=1e-2,
        e_weight=0.2
    )
    te_loss, te_acc = eval_ebt(
        ebt, test_loader, device,
        refine_steps=5, refine_lr=1e-2,
        e_weight=0.2
    )
    print(f"[EBT][Epoch {ep:02d}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | test_loss={te_loss:.4f} acc={te_acc:.4f}")

# Probe “thinking longer helps?”
print(">>> EBT: accuracy vs refinement steps (test set)")
for steps in [0, 1, 3, 4, 5, 8]:
    _, acc = eval_ebt(ebt, test_loader, device, refine_steps=steps, e_weight=0.2, refine_lr=5e-3)
    print(f"Refine steps = {steps:2d} -> test acc = {acc:.4f}")