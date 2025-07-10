const adata = {
    // X: main gene expression matrix (array or sparse matrix)
    // shape: (n_cells, n_genes)
    X: [
      // [gene1, gene2, gene3, geneN]
      [0,     3,     12,     0],      // cell1 expression
      [5,     0,     8,      1],      // cell2 expression
      [2,     1,     4,      0],      // cell3 expression
      // ...
    ],
  
    // obs: per-cell metadata (pandas dataframe)
    // shape: (n_cells, n_columns)
    obs: {
      'cell_1': {
        sample: 'ETV6-RUNX1_1',
        sample_type: 'ETV6-RUNX1',
        nGene: 2055,
        nUMI: 6759,
        percent_mito: 3.1,
        S_score: 0.13,
        G2M_score: 0.09,
        Phase: 'G1',
        UMAP1: 12.8,
        UMAP2: 5.1,
        tSNE_1: 22.3,
        tSNE_2: -3.7,
        celltype: 'B-cell',
        cluster: 'res.0.1_2'
      },
      'cell_2': {/** ... */},
      'cell_3': {/** ... */},
      // ...
    },
  
    // var: per-gene metadata (pandas dataframe)
    // shape: (n_genes, n_columns)
    var: {
      'CD3D': {
        gene_id: 'ENSG00000167286',
        n_cells: 31230
      },
      'RPL13A': {
        gene_id: 'ENSG00000142541',
        n_cells: 38900
      },
      'JUN': {
        gene_id: 'ENSG00000177606',
        n_cells: 23420
      },
      // ...
    },
  
    // obsm: multidimensional data per cell like UMAP, PCA, latent space embeddings (dict of arrays)
    // shape: (n_cells, n_dimensions)
    obsm: {
      X_umap: [
        [12.8, 5.1],    // cell1
        [11.3, 4.7],    // cell2
        [13.0, 5.3],    // cell3
        // ...
      ],
      X_scVI: [
        [0.12, -1.88, 0.45, /** ... */],  // cell1 (latent dim = 10–30)
        [0.02, -1.70, 0.32, /** ... */],  // cell2
        [0.10, -2.01, 0.50, /** ... */],  // cell3
        // ...
      ]
    },
  
    // uns: unstructured data like log1p, n_cells, n_genes, neighbors, etc. (dict)
    uns: {
      data_source: "Caron et al. (2020) - GSE132509",
      n_cells: 39375,
      n_genes: 25864,
      log1p: { base: 2 },
      neighbors: {
        connectivities: "...",
        params: {
          n_neighbors: 15,
          method: "umap"
        }
      }
    }
  }
  