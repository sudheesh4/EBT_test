# EBT on MNIST/Fashion-MNIST with a Frozen ViT Backbone

This repo contains a compact PyTorch implementation of an **Energy-Based head** trained on top of a **frozen Vision Transformer (ViT)** feature extractor. It showcases “**thinking longer helps**?” by iteratively refining a latent `z` via gradient descent on an energy function before producing logits.

* **Backbone:** any Hugging Face vision model (default: `facebook/dinov2-base`) kept **frozen**
* **Head:** EBT module with learnable initializer `g(r_x) → z0`, energy network `E(r_x, z)`, and classifier `h(z_T)`
* **Data:** MNIST or Fashion-MNIST auto-downloaded via `torchvision`
* **Loss:** multi-class hinge loss on energies (energies = `-logits`)
* **Probe:** quick sweep comparing test accuracy for different numbers of refinement steps

---

Traditional linear probes read the backbone’s representation once. Here:

1. **Initialize** a latent `z0` from the frozen representation `r_x`;
2. **Refine** `z` for `T` steps by descending `∂E/∂z`;
3. **Classify** with `h(z_T)`.

Can empirically check whether more refinement steps (i.e., more “thinking”) improves accuracy on the test set.

---

## Code map

* **Data & transforms**

  * `to_rgb_and_resize(img_size=224)`: Converts 1×28×28 digits to ImageNet-style 3×HxW with normalization.
  * `get_loaders(dataset, img_size, batch, workers)`: Returns train/test `DataLoader`s and number of classes.

* **Frozen ViT backbone**

  * `FrozenViTBackbone(...)`: Loads a Hugging Face vision model (default `facebook/dinov2-base`) and freezes it. Pools with either `CLS` or mean.

* **EBT head**

  * `EBTHead(embed_dim, num_classes, z_hidden)`:

    * `init_pred`: `r_x → z0` (linear or MLP)
    * `energy_net`: computes scalar energy `E(r_x, z)`
    * `to_logits`: maps refined `z_T` to class logits
    * `refine(...)`: runs `T` steps of gradient descent on `E` wrt `z`

* **Classifier wrapper**

  * `EBTClassifier(backbone, num_classes, z_hidden)`: Glues backbone + EBT head.

* **Loss**

  * `ebt_hinge_margin_loss(logits, targets, margin=1.0)`: multi-class hinge on **energies = -logits**.

* **Training & eval loops**

  * `train_ebt(...)`, `eval_ebt(...)`.

* **Experiment scaffold**

  * Trains for 5 epochs on Fashion-MNIST (batch 256, image size 224).
  * Evaluates test accuracy for several refinement steps to probe the “think longer” effect.

---

## Sample run on FashionMNIST 

EBT: accuracy vs refinement steps (test set) ===

** Un-trained model **

Refine steps =  0 -> test acc = 0.0772

Refine steps =  1 -> test acc = 0.0771

Refine steps =  3 -> test acc = 0.0772

Refine steps =  4 -> test acc = 0.0772

Refine steps =  5 -> test acc = 0.0772

Refine steps =  8 -> test acc = 0.0772

** After 5 epochs of training **

Refine steps =  0 -> test acc = 0.4642

Refine steps =  1 -> test acc = 0.6558

Refine steps =  3 -> test acc = 0.8667

Refine steps =  4 -> test acc = 0.8955

Refine steps =  5 -> test acc = 0.9088

Refine steps =  8 -> test acc = 0.9140
