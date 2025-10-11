# EBT on MNIST/Fashion-MNIST with a Frozen ViT Backbone

This repo contains a compact PyTorch implementation of an **Energy-Based head** (ref: https://energy-based-transformers.github.io/) trained on top of a **frozen Vision Transformer (ViT)** feature extractor. It showcases “**thinking longer helps**?” by iteratively refining a latent `z` via gradient descent on an energy function before producing logits.

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

----------------------------------------------------------------------------------------------------

## Sample run on FashionMNIST 

EBT: accuracy vs refinement steps (test set) ===

** Un-trained model **

Refine steps =  0 -> test acc = 0.0772

Refine steps =  1 -> test acc = 0.0771

Refine steps =  3 -> test acc = 0.0772

Refine steps =  4 -> test acc = 0.0772

Refine steps =  5 -> test acc = 0.0772

Refine steps =  8 -> test acc = 0.0772

-------------------------------------------------------------------

** After 5 epochs of training **

Refine steps =  0 -> test acc = 0.4642

Refine steps =  1 -> test acc = 0.6558

Refine steps =  3 -> test acc = 0.8667

Refine steps =  4 -> test acc = 0.8955

Refine steps =  5 -> test acc = 0.9088

Refine steps =  8 -> test acc = 0.9140

--------
##  Energy Network Contrastive Investigation

To evaluate whether the **EBT** head learned meaningful class-wise structure in the feature space, we conducted a *contrastive energy analysis* on the **FashionMNIST** test set, after training the head for 5 Epochs.

### Procedure

We fix a reference class `CLS` and extract:

* **Feature vectors** ( R_x = { r_x^{(i)} } ) from the frozen ViT backbone for samples belonging to that class.
* **Refined latent vectors** ( Z_S = { z_S^{(j)} } ) from the EBT model after `S` refinement steps, collected for *all* classes.

For each pair ((r_x^{(i)}, z_S^{(j)})), we compute the scalar energy
[
E(r_x^{(i)}, z_S^{(j)}),
]
and average the results per class of (z_S^{(j)}).
This yields an **average energy per class** indicating how compatible the feature representation of the chosen class is with the latent representations of every other class.

### Findings
 (LABEL KEYS => 0: T-shirt/top , 1: Trouser, 2: Pullover, 3: Dress, 4: Coat, 5: Sandal, 6: Shirt, 7: Sneaker, 8: Bag, 9: Ankle boot)
* **Same-class pairs** consistently exhibit **lower average energy** than cross-class pairs, confirming that the energy function learned a notion of intra-class compatibility in the joint ((r_x, z)) space.
* **Cross-class energies** are higher overall but **non-uniform**, some visually similar classes (e.g., Label 3 *Dress* vs. Label 0 *T-shirt/top*) remain closer in energy than more distinct ones (Label 5 *Sandal* vs. Label 2 *Pullover*).
* As the **number of refinement steps increases**, all energies shift **downward globally**, indicating that refinement progressively lowers the total energy landscape without proportionally widening inter-class gaps.

### Interpretation

These observations suggest that the current training regime primarily encourages the model to **minimize energy for correct (same-class) pairs**, while providing limited pressure to **raise energy for incorrect (cross-class) pairs**.
The network thus learns an attractive potential that aligns compatible representations but does not explicitly enforce repulsion between mismatched classes.

### Future Directions

To enhance class contrast and stabilize the energy scale, we could:

* Introduce **InfoNCE-style contrastive losses** to penalize low energies for negatives.
* Apply **batch-wise normalization** of energies to focus learning on relative rather than absolute scales.
* Incorporate **hard-negative sampling** to sharpen class boundaries in energy space.

