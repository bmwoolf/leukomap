# leukomap

Enhanced scRNA-seq Pipeline for Pediatric Leukemia

LeukoMap is a modern re-analysis of the Caron et al. (2020) pediatric leukemia single-cell RNA-seq dataset, from the [UConn scRNA class](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics).  
This project aims to extend the original analysis with deeper embeddings, clinical references, richer annotations, and functional drug mapping.

## Goals
- Understand intra-individual transcriptional heterogeneity in pediatric ALL
- Link developmental state to druggable biology
- Build a reusable, modular scRNA-seq pipeline for future leukemia engineers

## Root function
analyze(scRNA_seq_data, healthy_reference) -> annotated_clusters + druggable_targets

## Sub-functions
`load_data(data_dir) -> AnnData`  
`preprocess(AnnData) -> AnnData`  
`train_scvi(AnnData) -> scVI_model`  
`embed(scVI_model, AnnData) -> latent_space`  
`load_reference(reference_path) -> AnnData`  
`align_with_reference(latent_space, reference_AnnData) -> aligned_AnnData`  
`auto_annotate(aligned_AnnData) -> cell_type_labels`  
`compare_annotations(auto_labels, manual_labels) -> annotation_metrics`  
`launch_cellxgene(aligned_AnnData) -> interactive_dashboard`  
`run_pseudotime(aligned_AnnData) -> pseudotime_scores`  
`differential_expression(aligned_AnnData) -> DEG_table`  
`run_gsea(DEG_table) -> enriched_pathways`  
`query_lincs(DEG_table) -> druggable_targets`  
`export_outputs(AnnData, DEG_table, pathways, druggable_targets) -> annotated_clusters + druggable_targets files (CSV, JSON, PDF)`  
`package_pipeline(configs) -> Nextflow_workflow`   

---

## Data Sources
- [GEO: GSE132509](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE132509) — Caron et al. (2020)
- [Original paper](https://doi.org/10.1038/s41598-020-64929-x)
- [Class source](https://github.com/CBC-UCONN/Single-Cell-Transcriptomics)