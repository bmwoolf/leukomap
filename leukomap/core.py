"""
Core module for LeukoMap - Simplified base classes and interfaces.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
warnings.filterwarnings("ignore", category=FutureWarning)


class AnalysisStage(Enum):
    """Analysis stages."""
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    ANNOTATION = "annotation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    data_path: Optional[Path] = None
    output_dir: Path = Path("results")
    min_genes: int = 200
    min_cells: int = 3
    max_counts: Optional[int] = None
    max_genes: Optional[int] = None
    target_sum: int = 10000
    n_latent: int = 10
    n_hidden: int = 128
    n_layers: int = 2
    dropout_rate: float = 0.1
    batch_size: int = 128
    max_epochs: int = 400
    learning_rate: float = 1e-3
    batch_key: str = "sample_type"
    annotation_methods: List[str] = field(default_factory=lambda: ['celltypist'])
    confidence_threshold: float = 0.7
    resolution: float = 0.5
    n_neighbors: int = 15
    n_pcs: int = 50
    figure_dpi: int = 300
    figure_format: str = 'png'
    gpu_id: int = 0  # GPU device ID
    
    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        for subdir in ['data', 'models', 'results', 'figures', 'reports']:
            (self.output_dir / subdir).mkdir(exist_ok=True)


class BaseProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def save_results(self, data: Any, output_path: Path) -> None:
        pass
    
    def validate_input(self, data: Any) -> bool:
        return True
    
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.ANALYSIS


class DataProcessor(BaseProcessor):
    """Base class for data processing operations."""
    
    def __init__(self, config: AnalysisConfig):
        super().__init__(config)
        self._adata: Optional[ad.AnnData] = None
    
    @property
    def adata(self) -> Optional[ad.AnnData]:
        return self._adata
    
    @adata.setter
    def adata(self, value: ad.AnnData):
        self._adata = value
    
    def validate_input(self, data: ad.AnnData) -> bool:
        if not isinstance(data, ad.AnnData):
            self.logger.error("Input must be an AnnData object")
            return False
        if data.n_obs == 0 or data.n_vars == 0:
            self.logger.error("AnnData object is empty")
            return False
        return True
    
    def save_results(self, data: ad.AnnData, output_path: Path) -> None:
        if not self.validate_input(data):
            raise ValueError("Invalid AnnData object")
        data.write(output_path)
        self.logger.info(f"Saved AnnData to {output_path}")


class Pipeline:
    """Main analysis pipeline that orchestrates multiple processors."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.processors: List[BaseProcessor] = []
        self.results: Dict[str, Any] = {}
        self.current_stage = AnalysisStage.DATA_LOADING
    
    def add_processor(self, processor: BaseProcessor) -> 'Pipeline':
        self.processors.append(processor)
        return self
    
    def run(self, initial_data: Optional[Any] = None) -> Dict[str, Any]:
        self.logger.info("Starting pipeline execution")
        current_data = initial_data
        
        for i, processor in enumerate(self.processors):
            try:
                self.logger.info(f"Running processor {i+1}/{len(self.processors)}: {processor.__class__.__name__}")
                current_data = processor.process(current_data)
                self.results[processor.__class__.__name__] = current_data
            except Exception as e:
                self.logger.error(f"Processor {processor.__class__.__name__} failed: {e}")
                raise
        
        self.logger.info("Pipeline execution complete")
        return self.results
    
    def get_results(self, stage: Optional[AnalysisStage] = None) -> Any:
        if stage is None:
            return self.results
        return {k: v for k, v in self.results.items() if hasattr(v, 'get_stage') and v.get_stage() == stage}
    
    def save_all_results(self) -> None:
        for processor in self.processors:
            if hasattr(processor, 'adata') and processor.adata is not None:
                output_path = self.config.output_dir / 'data' / f"{processor.__class__.__name__.lower()}_results.h5ad"
                processor.save_results(processor.adata, output_path)


class ResultTracker:
    """Simple result tracking."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
    
    def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.results[key] = data
        if metadata:
            self.metadata[key] = metadata
    
    def get(self, key: str) -> Any:
        return self.results.get(key)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        return self.metadata.get(key, {})
    
    def save_to_disk(self, key: str, filename: Optional[str] = None) -> Path:
        if key not in self.results:
            raise KeyError(f"Result '{key}' not found")
        
        if filename is None:
            filename = f"{key}.h5ad"
        
        output_path = self.output_dir / filename
        
        data = self.results[key]
        if isinstance(data, ad.AnnData):
            data.write(output_path)
        else:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        return output_path
    
    def list_results(self) -> List[str]:
        return list(self.results.keys())


class LeukoMapAnalysis:
    """Main analysis class."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.pipeline = Pipeline(config)
        self.tracker = ResultTracker(config.output_dir)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run_full_analysis_with_data(self, adata, healthy_reference=None):
        """Run full analysis with provided data."""
        self.logger.info("Starting full LeukoMap analysis")
        
        # Store original data
        self.tracker.store('original_data', adata)
        
        # Step 1: Preprocessing
        self.logger.info("Step 1: Preprocessing")
        from .preprocessing import PreprocessingPipeline
        preprocessor = PreprocessingPipeline(self.config)
        adata = preprocessor.process(adata)
        self.tracker.store('preprocessed_data', adata)

        # === Diagnostics: Check AnnData for NaNs, Infs, negatives, all-zero cells/genes ===
        import numpy as np
        
        # Check scaled data (adata.X) - can have negatives for PCA/UMAP
        X_scaled = adata.X
        if hasattr(X_scaled, 'toarray'):
            X_scaled = X_scaled.toarray()
        nan_count_scaled = np.isnan(X_scaled).sum()
        inf_count_scaled = np.isinf(X_scaled).sum()
        neg_count_scaled = (X_scaled < 0).sum()
        
        self.logger.info(f"[Diagnostics] Scaled data (adata.X): shape={X_scaled.shape}")
        self.logger.info(f"[Diagnostics] Scaled data - NaN: {nan_count_scaled}, Inf: {inf_count_scaled}, Neg: {neg_count_scaled}")
        self.logger.info(f"[Diagnostics] Scaled data - min={np.nanmin(X_scaled):.3f}, max={np.nanmax(X_scaled):.3f}, mean={np.nanmean(X_scaled):.3f}")
        
        # Check raw data (adata.raw.X) - should be non-negative for scVI
        if hasattr(adata, 'raw') and adata.raw is not None:
            X_raw = adata.raw.X
            if hasattr(X_raw, 'toarray'):
                X_raw = X_raw.toarray()
            nan_count_raw = np.isnan(X_raw).sum()
            inf_count_raw = np.isinf(X_raw).sum()
            neg_count_raw = (X_raw < 0).sum()
            zero_cells_raw = (X_raw.sum(axis=1) == 0).sum()
            zero_genes_raw = (X_raw.sum(axis=0) == 0).sum()
            
            self.logger.info(f"[Diagnostics] Raw data (adata.raw.X): shape={X_raw.shape}")
            self.logger.info(f"[Diagnostics] Raw data - NaN: {nan_count_raw}, Inf: {inf_count_raw}, Neg: {neg_count_raw}")
            self.logger.info(f"[Diagnostics] Raw data - Zero cells: {zero_cells_raw}, Zero genes: {zero_genes_raw}")
            self.logger.info(f"[Diagnostics] Raw data - min={np.nanmin(X_raw):.3f}, max={np.nanmax(X_raw):.3f}, mean={np.nanmean(X_raw):.3f}")
            
            # Check for issues in raw data (used for scVI)
            if nan_count_raw > 0 or inf_count_raw > 0 or neg_count_raw > 0 or zero_cells_raw > 0 or zero_genes_raw > 0:
                raise ValueError(f"[Diagnostics] Raw data issue detected: NaN={nan_count_raw}, Inf={inf_count_raw}, Neg={neg_count_raw}, Zero cells={zero_cells_raw}, Zero genes={zero_genes_raw}. Aborting before scVI training.")
            else:
                self.logger.info("[Diagnostics] Raw data is clean - ready for scVI training")
        else:
            self.logger.warning("[Diagnostics] No adata.raw found - will use adata.X for scVI")
            # Check scaled data for scVI if no raw data
            if nan_count_scaled > 0 or inf_count_scaled > 0:
                raise ValueError(f"[Diagnostics] Scaled data issue detected: NaN={nan_count_scaled}, Inf={inf_count_scaled}. Aborting before scVI training.")
        
        # === End diagnostics ===
        
        # Step 2: scVI Training
        self.logger.info("Step 2: scVI Training")
        from .scvi_training import SCVITrainer
        trainer = SCVITrainer(self.config)
        scvi_model = trainer.process(adata)
        self.tracker.store('scvi_model', scvi_model)
        
        # Step 3: Generate embeddings
        self.logger.info("Step 3: Generating embeddings")
        latent_space = trainer.generate_embeddings(scvi_model, adata)
        self.tracker.store('latent_space', latent_space)
        
        # Step 4: Load and align with healthy reference
        if healthy_reference:
            self.logger.info("Step 4: Aligning with healthy reference")
            reference_adata = self.load_reference(healthy_reference)
            aligned_adata = self.align_with_reference(latent_space, reference_adata)
            self.tracker.store('aligned_data', aligned_adata)
            adata = aligned_adata
        else:
            self.logger.info("Step 4: No healthy reference provided, skipping alignment")
        
        # Step 5: Cell type annotation
        self.logger.info("Step 5: Auto cell type annotation")
        try:
            # Use our new auto cell type labeling system
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
            from auto_celltype_labeling import AutoCellTypeLabeler
            
            # Create temporary file for annotation
            temp_adata_path = self.config.output_dir / 'data' / 'temp_for_annotation.h5ad'
            adata.write(temp_adata_path)
            
            # Run auto annotation
            labeler = AutoCellTypeLabeler(output_dir=str(self.config.output_dir / 'annotations'))
            annotation_results = labeler.run_annotation_pipeline(str(temp_adata_path))
            
            # Add annotations to adata
            if not annotation_results.empty:
                # Create a mapping from cell_id to predicted_celltype
                cell_type_map = dict(zip(annotation_results['cell_id'], annotation_results['predicted_celltype']))
                confidence_map = dict(zip(annotation_results['cell_id'], annotation_results['confidence']))
                
                # Add to adata.obs
                adata.obs['predicted_celltype'] = [cell_type_map.get(cell_id, 'Unknown') for cell_id in adata.obs_names]
                adata.obs['annotation_confidence'] = [confidence_map.get(cell_id, 0.0) for cell_id in adata.obs_names]
                adata.obs['annotation_method'] = annotation_results['method'].iloc[0]
                
                self.logger.info(f"Auto annotation completed: {annotation_results['predicted_celltype'].nunique()} cell types identified")
            else:
                self.logger.warning("Auto annotation failed, using fallback")
                adata.obs['predicted_celltype'] = 'Unknown'
                adata.obs['annotation_confidence'] = 0.0
                adata.obs['annotation_method'] = 'fallback'
            
            # Clean up temp file
            if temp_adata_path.exists():
                temp_adata_path.unlink()
                
        except Exception as e:
            self.logger.warning(f"Auto annotation failed: {e}, using fallback")
            adata.obs['predicted_celltype'] = 'Unknown'
            adata.obs['annotation_confidence'] = 0.0
            adata.obs['annotation_method'] = 'fallback'
        
        self.tracker.store('annotated_data', adata)
        
        # Step 6: Advanced analysis (pseudotime, GSEA, drug targets)
        self.logger.info("Step 6: Advanced analysis pipeline")
        
        # Run pseudotime analysis
        self.logger.info("Step 6a: Pseudotime analysis")
        adata = self.run_pseudotime_analysis(adata)
        
        # Run differential expression
        self.logger.info("Step 6b: Differential expression analysis")
        deg_results = self.run_differential_expression(adata)
        self.tracker.store('differential_expression', deg_results)
        
        # Run GSEA analysis
        self.logger.info("Step 6c: Gene Set Enrichment Analysis")
        gsea_results = self.run_python_gsea(adata)
        self.tracker.store('gsea_results', gsea_results)
        
        # Run drug target identification
        self.logger.info("Step 6d: Drug target identification")
        drug_targets = self.query_drug_databases(deg_results)
        self.tracker.store('druggable_targets', drug_targets)
        
        # Step 7: Generate visualizations
        self.logger.info("Step 7: Generating visualizations")
        try:
            from scripts.visualization_pipeline import LeukoMapVisualizer
            visualizer = LeukoMapVisualizer(str(self.config.output_dir / 'figures'))
            figures = visualizer.create_comprehensive_report(adata, 'leukomap')
            self.tracker.store('visualization_figures', figures)
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
        
        # Step 8: Export results
        self.logger.info("Step 8: Exporting results")
        export_paths = self.export_results(adata, deg_results, gsea_results, drug_targets)
        self.tracker.store('export_paths', export_paths)
        
        # Generate analysis report
        report_path = self.save_analysis_report()
        self.tracker.store('analysis_report', report_path)
        
        self.logger.info("Full analysis complete")
        
        return {
            'annotated_data': adata,
            'druggable_targets': drug_targets,
            'differential_expression': deg_results,
            'analysis_report': report_path,
            'all_results': self.tracker.results
        }
    
    def run_full_analysis(self, data_path: Optional[Path] = None) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        if data_path:
            self.config.data_path = data_path
        
        self.logger.info("Starting full LeukoMap analysis")
        
        # Add processors to pipeline
        from .data_loading import DataLoader
        from .preprocessing import PreprocessingPipeline
        
        self.pipeline.add_processor(DataLoader(self.config))
        self.pipeline.add_processor(PreprocessingPipeline(self.config))
        
        # Run pipeline
        results = self.pipeline.run()
        
        # Store results
        for key, value in results.items():
            self.tracker.store(key, value)
        
        self.logger.info("Full analysis complete")
        return results
    
    def load_reference(self, reference_path):
        """Load healthy reference data."""
        if isinstance(reference_path, (str, Path)):
            import scanpy as sc
            return sc.read_h5ad(reference_path)
        else:
            return reference_path
    
    def align_with_reference(self, latent_space, reference_adata):
        """Align data with healthy reference."""
        import scanpy as sc
        
        # Add reference information to latent space
        latent_space.obs['is_reference'] = False
        reference_adata.obs['is_reference'] = True
        
        # Concatenate data
        aligned_adata = sc.concat([latent_space, reference_adata], join='outer')
        
        self.logger.info(f"Aligned data: {aligned_adata.n_obs} cells, {aligned_adata.n_vars} genes")
        return aligned_adata
    
    def run_pseudotime_analysis(self, adata):
        """Run pseudotime analysis."""
        try:
            import scvelo as scv
            import scanpy as sc
            
            # Basic pseudotime analysis
            sc.pp.neighbors(adata, use_rep='X_scVI')
            sc.tl.umap(adata)
            sc.tl.leiden(adata)
            
            # Compute velocity (if velocity data available)
            if 'velocity' in adata.layers:
                scv.tl.velocity_graph(adata)
                scv.tl.velocity_pseudotime(adata)
            
            self.logger.info("Pseudotime analysis completed")
            return adata
            
        except ImportError:
            self.logger.warning("scVelo not available, skipping pseudotime analysis")
            return adata
    
    def run_differential_expression(self, adata):
        """Run differential expression analysis."""
        import scanpy as sc
        import pandas as pd
        
        # Ensure we have clustering
        if 'leiden' not in adata.obs.columns:
            sc.pp.neighbors(adata, use_rep='X_scVI')
            sc.tl.leiden(adata)
        
        # Run differential expression
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
        
        # Extract results
        deg_results = {}
        for cluster in adata.obs['leiden'].unique():
            cluster_genes = sc.get.rank_genes_groups_df(adata, group=cluster)
            deg_results[cluster] = cluster_genes
        
        self.logger.info(f"Differential expression analysis completed for {len(deg_results)} clusters")
        return deg_results
    
    def run_python_gsea(self, adata):
        """Run GSEA using Python alternatives to GSVA/fgsea."""
        try:
            import gseapy as gp
            import scanpy as sc
            
            gsea_results = {}
            
            # Get differential expression results
            if 'predicted_celltype' in adata.obs.columns:
                sc.tl.rank_genes_groups(adata, 'predicted_celltype', method='wilcoxon')
                
                for cell_type in adata.obs['predicted_celltype'].unique():
                    if cell_type != 'Unknown':
                        # Get top genes for this cell type
                        genes_df = sc.get.rank_genes_groups_df(adata, group=cell_type)
                        top_genes = genes_df.head(100)['names'].tolist()
                        
                        # Run GSEA
                        enr = gp.enrichr(
                            gene_list=top_genes,
                            gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021', 'Reactome_2022'],
                            organism='Human',
                            outdir=None
                        )
                        
                        gsea_results[cell_type] = enr.results
            
            self.logger.info(f"GSEA analysis completed for {len(gsea_results)} cell types")
            return gsea_results
            
        except ImportError:
            self.logger.warning("gseapy not available, using simple pathway enrichment")
            return {}
    
    def run_gsea_analysis(self, deg_table):
        """Legacy method - use run_python_gsea instead."""
        return self.run_python_gsea(deg_table)
    
    def query_drug_databases(self, deg_results):
        """Query drug databases using Python APIs."""
        try:
            import requests
            import json
            
            drug_results = {}
            
            for cell_type, genes_df in deg_results.items():
                if len(genes_df) > 0:
                    # Get top upregulated genes
                    top_genes = genes_df[genes_df['logfoldchanges'] > 0].head(50)['names'].tolist()
                    
                    # Query Enrichr
                    enr_url = "https://maayanlab.cloud/Enrichr/addList"
                    payload = {
                        'list': (None, '\n'.join(top_genes)),
                        'description': (None, f'{cell_type}_upregulated')
                    }
                    
                    response = requests.post(enr_url, files=payload)
                    if response.status_code == 200:
                        user_list_id = response.json()['userListId']
                        
                        # Get results
                        results_url = f"https://maayanlab.cloud/Enrichr/enrich?userListId={user_list_id}&backgroundType=LINCS_L1000_Chem_Pert_up"
                        response = requests.get(results_url)
                        if response.status_code == 200:
                            drug_results[cell_type] = response.json()
            
            # Add simple drug associations
            drug_gene_associations = self._simple_drug_associations(deg_results)
            drug_results['drug_gene_associations'] = drug_gene_associations
            
            self.logger.info("Drug database query completed")
            return drug_results
            
        except Exception as e:
            self.logger.error(f"Drug database query failed: {e}")
            return self._simple_drug_associations(deg_results)
    
    def query_lincs_database(self, deg_table):
        """Legacy method - use query_drug_databases instead."""
        return self.query_drug_databases(deg_table)
    
    def _simple_drug_associations(self, deg_results):
        """Simple drug-gene associations without external APIs."""
        drug_associations = {}
        for cell_type, genes_df in deg_results.items():
            if len(genes_df) > 0:
                top_genes = genes_df.head(10)['names'].tolist()
                drug_associations[cell_type] = {
                    'top_genes': top_genes,
                    'potential_drugs': ['Drug_A', 'Drug_B', 'Drug_C']  # Mock drugs
                }
        return drug_associations
    
    def export_results(self, adata, deg_table, pathways, druggable_targets):
        """Export analysis outputs."""
        import pandas as pd
        import json
        
        export_paths = {}
        
        # Export annotated data
        adata_path = self.config.output_dir / 'annotated_data.h5ad'
        adata.write(adata_path)
        export_paths['annotated_data'] = str(adata_path)
        
        # Export differential expression results
        deg_path = self.config.output_dir / 'differential_expression.csv'
        deg_df = pd.concat(deg_table.values(), keys=deg_table.keys())
        deg_df.to_csv(deg_path)
        export_paths['differential_expression'] = str(deg_path)
        
        # Export GSEA results
        if pathways:
            gsea_path = self.config.output_dir / 'gsea_results.json'
            with open(gsea_path, 'w') as f:
                json.dump(pathways, f, indent=2)
            export_paths['gsea_results'] = str(gsea_path)
        
        # Export drug targets
        if druggable_targets:
            drug_path = self.config.output_dir / 'druggable_targets.json'
            with open(drug_path, 'w') as f:
                json.dump(druggable_targets, f, indent=2)
            export_paths['druggable_targets'] = str(drug_path)
        
        self.logger.info(f"Results exported to {self.config.output_dir}")
        return export_paths
    
    def get_adata(self) -> Optional[ad.AnnData]:
        return self.tracker.get('annotated_data')
    
    def save_analysis_report(self) -> Path:
        report_path = self.config.output_dir / 'reports' / 'analysis_report.txt'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("LeukoMap Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis completed: {len(self.tracker.list_results())} stages\n")
            f.write(f"Results available: {', '.join(self.tracker.list_results())}\n")
            
            # Add summary statistics
            adata = self.get_adata()
            if adata is not None:
                f.write(f"\nData Summary:\n")
                f.write(f"- Cells: {adata.n_obs}\n")
                f.write(f"- Genes: {adata.n_vars}\n")
                f.write(f"- Cell types: {len(adata.obs['celltypist_cell_type'].unique()) if 'celltypist_cell_type' in adata.obs.columns else 'N/A'}\n")
        
        return report_path 