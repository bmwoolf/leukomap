# leukomap

Enhanced scRNA-seq Pipeline for Pediatric Leukemia

LeukoMap is a modern re-analysis of the Caron et al. (2020) pediatric leukemia single-cell RNA-seq dataset, from the [UConn scRNA class](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics).  
This project aims to extend the original analysis with deeper embeddings, clinical references, richer annotations, and functional drug mapping.

## Goals
- Understand intra-individual transcriptional heterogeneity in pediatric ALL
- Link developmental state to druggable biology
- Build a reusable, modular scRNA-seq pipeline for future leukemia engineers

## Planned Pipeline Tasks

### Data Ingestion & Embedding
- [ ] Load Caron et al. scRNA-seq data into `AnnData` format
- [ ] Train a latent model using **scVI** or **scANVI** for clustering
- [ ] Visualize embeddings with UMAP
- [ ] Integrate a **healthy bone marrow reference** (e.g. Human Cell Atlas) to detect abnormal clusters

### Annotation & Exploration
- [ ] Run **Azimuth** (Seurat) or **SingleR** (R) for auto cell type labeling
- [ ] Compare auto vs manual annotations (e.g. ARI/NMI)
- [ ] Load into **Cellxgene** or **Cirro** for interactive visualization
- [ ] Run **pseudotime analysis** with Monocle3 or scVelo to trace leukemia progression

### Functional Analysis
- [ ] Perform **differential expression** between clusters
- [ ] Run **single-cell GSEA** using `GSVA`, `fgsea`, or `ssGSEA`
- [ ] Link DEGs to perturbation signatures via **LINCS1000** / Enrichr
- [ ] Highlight top candidate pathways and drugs per cluster

### Workflow Packaging
- [ ] Wrap analysis in **Nextflow** with **Docker/Conda** for portability
- [ ] Use **nf-core** or minimal `Snakefile` template
- [ ] Export results as **CSV/JSON**, ready for ML or ontology integration
- [ ] Auto-generate publication-quality figures and PDF/PowerPoint summaries

---

## Data Sources
- [GEO: GSE132509](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132509) — Caron et al. (2020)
- [Original paper](https://doi.org/10.1038/s41598-020-64929-x)
- [Class source](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics)