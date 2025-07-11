# LeukoMap: scVI Latent Space Analysis of Pediatric Leukemia

## Dataset
- **39,211 cells** from Caron et al. (2020) pediatric ALL dataset
- **23,198 genes** analyzed across 4 sample types
- **10-dimensional latent representation** learned by scVI

## Sample Distribution
- ETV6-RUNX1: 17,936 cells (45.7%) - primary leukemia subtype
- HHD: 8,634 cells (22.0%) - healthy donor controls  
- PBMMC: 6,939 cells (17.7%) - peripheral blood mononuclear cells
- PRE-T: 5,702 cells (14.5%) - pre-T cell samples

## Cell Type Clusters
scVI discovered **11 distinct cell populations**:
- ETV6.RUNX1.1-4: 4 leukemia-specific subtypes (34.2% of cells)
- HHD.1-2: 2 healthy donor subtypes (19.9% of cells)
- Erythrocytes: 4,604 cells (11.7%)
- T cells + NK: 3,985 cells (10.2%)
- B cells + Mono: 2,849 cells (7.3%)
- PRE-T.1-2: 2 pre-T subtypes (11.8% of cells)

## Quality Metrics
- Mean UMIs per cell: 5,368 ± 3,912
- Mean genes per cell: 1,644 ± 781
- Mitochondrial content: <0.06% (excellent quality)

## Key Findings
scVI successfully separated leukemia subtypes from healthy controls in latent space. The 10-dimensional representation captured biological variation while removing technical noise, enabling clear visualization of disease-specific transcriptional states.

**Next steps**: Differential expression analysis between ETV6-RUNX1 subtypes to identify druggable targets. 