![Banner](assets/github_banner.png)

# LeukoMap

Enhanced scRNA-seq Pipeline for Pediatric Leukemia

LeukoMap is a modern re-analysis of the Caron et al. (2020) pediatric leukemia single-cell RNA-seq dataset, from the [UConn scRNA class](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics). This project aims to extend the original analysis with deeper embeddings, clinical references, richer annotations, and functional drug mapping.

## Goals
- Understand intra-individual transcriptional heterogeneity in pediatric ALL
- Link developmental state to druggable biology
- Build a reusable, modular scRNA-seq pipeline for future leukemia engineers

## Root function
`analyze(scRNA_seq_data, healthy_reference) -> annotated_clusters + druggable_targets`

-> `load_data(data_dir) -> AnnData`  
-> `preprocess(AnnData) -> AnnData`  
-> `train_scvi(AnnData) -> scVI_model`  
-> `embed(scVI_model, AnnData) -> latent_space`  
-> `load_reference(reference_path) -> AnnData`  
-> `align_with_reference(latent_space, reference_AnnData) -> aligned_AnnData`  
-> `auto_annotate(aligned_AnnData) -> cell_type_labels`  
-> `compare_annotations(auto_labels, manual_labels) -> annotation_metrics`  
-> `launch_cellxgene(aligned_AnnData) -> interactive_dashboard`  
-> `run_pseudotime(aligned_AnnData) -> pseudotime_scores`  
-> `differential_expression(aligned_AnnData) -> DEG_table`  
-> `run_gsea(DEG_table) -> enriched_pathways`  
-> `query_lincs(DEG_table) -> druggable_targets`  
-> `export_outputs(AnnData, DEG_table, pathways, druggable_targets) -> annotated_clusters + druggable_targets files (CSV, JSON, PDF)`  
-> `package_pipeline(configs) -> Nextflow_workflow`   

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bmwoolf/leukomap.git
cd leukomap
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Pretrained Model Weights

Pretrained scVI model weights for LeukoMap are available on Hugging Face Hub:
https://huggingface.co/bmwoolf/leukomap-scvi

You can load the model directly in your code as follows:
```python
from scvi.model import SCVI
import scanpy as sc

# Load your AnnData object
adata = sc.read_h5ad("your_data.h5ad")

# Load the pretrained model from Hugging Face
model = SCVI.load("bmwoolf/leukomap-scvi", adata)

# Get latent representation
latent = model.get_latent_representation()
```

## AnnData Object
Please see `./adata/adata_example.js` and `./adata/adata_example.py` for example outlines of how you would represent real world cell and gene annotations in memory. Pretty cool.

---
## Running the code

1. Clone the repository:
```bash
git clone https://github.com/bmwoolf/leukomap.git
cd leukomap
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Sources
- [GEO: GSE132509](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132509) — Caron et al. (2020)
- [Original paper](https://doi.org/10.1038/s41598-020-64929-x)
- [Class source](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics)