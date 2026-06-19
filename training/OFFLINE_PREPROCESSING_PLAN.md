Short answer: **GPU caching doesn't scale past ~hundreds of scenes**, but the *underlying insight* — that preprocessing is pure repeated computation on fixed data — does scale, just via a different mechanism.

Here's how the caching strategy should evolve at each tier:

---

### Current (10 scenes) → GPU cache ✅
Everything fits in VRAM. Cache preprocessed tensors directly on device at startup. Ideal.

### Small scale (~100–2k scenes) → CPU RAM cache
Too big for VRAM, but fits in CPU RAM (128–512 GB on a typical server). Cache `_preprocess_batch` outputs in a Python list on CPU, use `DataLoader(pin_memory=True)` to stream batches to GPU efficiently. Still avoids all disk I/O and recomputation.

### Large scale (10k–100k+ scenes) → **Offline preprocessing (disk cache)**
This is the approach that scales to any size, and it's arguably the *right* long-term investment regardless:

The expensive work in `_preprocess_batch` — coord transforms, ray generation, texture log-encoding — are **all deterministic functions of fixed scene geometry and cameras**. They never change between epochs. So the real fix is to precompute them *once* and save them back to disk alongside the H5 files. The DataLoader then just loads pre-baked tensors with no math at runtime.

Concretely, this means a one-time `preprocess_dataset.py` script that runs over your data dir, computes `rays_o`, `rays_d`, `tri_vpos_view_tf`, `texture_log`, etc. for each scene, and appends them to the H5 file (or writes a companion `.pt` file). Then `SceneDataset.__getitem__` just reads and returns them — no transforms at training time.

```
# H5 layout after offline preprocessing:
scene_0001.h5
  ├── triangles          # raw (already there)
  ├── texture            # raw (already there)
  ├── gt_img             # raw (already there)
  ├── texture_log        # precomputed ← new
  ├── rays_o             # precomputed ← new
  ├── rays_d             # precomputed ← new
  └── tri_vpos_view_tf   # precomputed ← new
```

This strategy also works perfectly with your current 10-scene setup and grows linearly with dataset size with no code changes to the training loop.

---

### Recommendation for you

| Now | Phase 2 (scale-up) |
|---|---|
| GPU cache for fast iteration | Write a `preprocess_dataset.py` script offline |
| `torch.compile` + AMP bf16 | Load pre-baked tensors from H5 in `SceneDataset` |
| Batch size = full dataset | `num_workers=4`, `pin_memory=True` |

The GPU cache is still worth doing *today* for the iteration speed boost. But when you're ready to scale up, the offline preprocessing approach gives you the same benefit (eliminating per-epoch recomputation) without any memory constraints.