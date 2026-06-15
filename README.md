# ReguSync

**ReguSync** is a GRN-guided single-cell multimodal language model for cross-modal translation in single-cell and spatial multi-omics data.

Single-cell multi-omics profiling provides powerful insights into cellular heterogeneity, but it is often costly, technically noisy, and incomplete across modalities. ReguSync addresses these challenges by leveraging gene regulatory networks (GRNs) to guide both intra-modal feature modeling and cross-modal semantic synchronization.

## Overview

ReguSync integrates the modeling capacity of single-cell language models with regulatory knowledge derived from GRNs. Within each modality, regulatory logic guides self-attention toward biologically meaningful dependencies. Across modalities, ReguSync learns a shared biological context from the GRN and uses cross-attention to synchronize semantic representations across omics layers.

## Key Features

* GRN-guided modeling for single-cell multi-omics data
* Cross-modal translation between missing omics profiles
* Regulatory logic-guided self-attention within modalities
* GRN-derived biological context for cross-modal semantic synchronization
* Robust generalization to spatial multi-omics datasets
* Support for downstream analyses, including:

  * cell-type annotation
  * pseudotime inference
  * perturbation response prediction

## Applications

ReguSync can be applied to single-cell and spatial multi-omics translation tasks, including the inference of missing transcriptomic, chromatin accessibility, or protein profiles from paired or partially observed multi-omics data.

## Installation

```bash
git clone https://github.com/your-username/ReguSync.git
cd ReguSync
pip install -r requirements.txt
```

## Quick Start

```python
# Example usage
from regusync import ReguSync

model = ReguSync()
model.train(data)
predicted_profiles = model.translate(query_data)
```

## Citation

If you use ReguSync in your research, please cite our work:

```bibtex
@article{ReguSync,
  title   = {ReguSync: A GRN-guided single-cell multimodal language model for cross-modal translation},
  author  = {Your Name et al.},
  journal = {To be updated},
  year    = {2026}
}
```

## License

This project is released under the MIT License.
