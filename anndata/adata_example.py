import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix

# -------------------------------
# X: Gene expression matrix
# -------------------------------
X = np.array([
    [0, 3, 12, 0],    # cell1
    [5, 0, 8, 1],     # cell2
    [2, 1, 4, 0],     # cell3
])
# Or sparse: X = csr_matrix(X)

# -------------------------------
# obs: Per-cell metadata
# -------------------------------
obs = pd.DataFrame({
    'sample': ['ETV6-RUNX1_1', 'ETV6-RUNX1_1', 'ETV6-RUNX1_1'],
    'sample_type': ['ETV6-RUNX1'] * 3,
    'nGene': [2055, 2100, 2001],
    'nUMI': [6759, 7001, 6123],
    'percent_mito': [3.1, 2.8, 3.5],
    'S_score': [0.13, 0.10, 0.15],
    'G2M_score': [0.09, 0.11, 0.08],
    'Phase': ['G1', 'G1', 'S'],
    'UMAP1': [12.8, 11.3, 13.0],
    'UMAP2': [5.1, 4.7, 5.3],
    'tSNE_1': [22.3, 21.1, 23.0],
    'tSNE_2': [-3.7, -4.0, -3.5],
    'celltype': ['B-cell', 'B-cell', 'B-cell'],
    'cluster': ['res.0.1_2', 'res.0.1_2', 'res.0.1_3']
}, index=['cell_1', 'cell_2', 'cell_3'])

# -------------------------------
# var: Per-gene metadata
# -------------------------------
var = pd.DataFrame({
    'gene_id': [
        'ENSG00000167286',  # CD3D
        'ENSG00000142541',  # RPL13A
        'ENSG00000177606',  # JUN
        'ENSG00000111111',  # DummyGene
    ],
    'n_cells': [31230, 38900, 23420, 11900]
}, index=['CD3D', 'RPL13A', 'JUN', 'DUMMY'])

# -------------------------------
# obsm: Multi-dimensional arrays per cell
# -------------------------------
obsm = {
    'X_umap': np.array([
        [12.8, 5.1],
        [11.3, 4.7],
        [13.0, 5.3]
    ]),
    'X_scVI': np.array([
        [0.12, -1.88, 0.45],
        [0.02, -1.70, 0.32],
        [0.10, -2.01, 0.50]
    ])
}

# -------------------------------
# uns: Unstructured metadata
# -------------------------------
uns = {
    'data_source': 'Caron et al. (2020) - GSE132509',
    'n_cells': 39375,
    'n_genes': 25864,
    'log1p': {'base': 2},
    'neighbors': {
        'connectivities': '...',
        'params': {
            'n_neighbors': 15,
            'method': 'umap'
        }
    }
}

adata = ad.AnnData(
    X=X,
    obs=obs,
    var=var,
    obsm=obsm,
    uns=uns
)
