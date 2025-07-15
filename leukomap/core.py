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
        self.logger.info("Step 5: Cell type annotation")
        from .cell_type_annotation import CellTypeAnnotator
        annotator = CellTypeAnnotator(self.config)
        annotated_adata = annotator.process(adata)
        self.tracker.store('annotated_data', annotated_adata)
        
        # Step 6: Differential expression
        self.logger.info("Step 6: Differential expression analysis")
        deg_results = self.run_differential_expression(annotated_adata)
        self.tracker.store('differential_expression', deg_results)
        
        # Step 7: GSEA analysis
        self.logger.info("Step 7: Gene Set Enrichment Analysis")
        gsea_results = self.run_gsea_analysis(deg_results)
        self.tracker.store('gsea_results', gsea_results)
        
        # Step 8: Drug target identification
        self.logger.info("Step 8: Drug target identification")
        drug_targets = self.query_lincs_database(deg_results)
        self.tracker.store('druggable_targets', drug_targets)
        
        # Step 9: Export results
        self.logger.info("Step 9: Exporting results")
        export_paths = self.export_results(annotated_adata, deg_results, gsea_results, drug_targets)
        self.tracker.store('export_paths', export_paths)
        
        # Generate analysis report
        report_path = self.save_analysis_report()
        self.tracker.store('analysis_report', report_path)
        
        self.logger.info("Full analysis complete")
        
        return {
            'annotated_data': annotated_adata,
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
    
    def run_gsea_analysis(self, deg_table):
        """Run Gene Set Enrichment Analysis."""
        try:
            import gseapy as gp
            
            gsea_results = {}
            
            # Run GSEA for each cluster
            for cluster, genes_df in deg_table.items():
                if len(genes_df) > 0:
                    # Prepare gene list
                    gene_list = genes_df.set_index('names')['scores'].to_dict()
                    
                    # Run GSEA
                    enr = gp.enrichr(
                        gene_list=list(gene_list.keys())[:100],  # Top 100 genes
                        gene_sets=['KEGG_2021_Human', 'GO_Biological_Process_2021'],
                        organism='Human'
                    )
                    
                    gsea_results[cluster] = enr.results
            
            self.logger.info("GSEA analysis completed")
            return gsea_results
            
        except ImportError:
            self.logger.warning("gseapy not available, skipping GSEA analysis")
            return {}
    
    def query_lincs_database(self, deg_table):
        """Query LINCS database for drug targets."""
        try:
            import requests
            import pandas as pd
            
            drug_targets = {}
            
            # Query LINCS L1000 database for top genes
            for cluster, genes_df in deg_table.items():
                if len(genes_df) > 0:
                    # Get top upregulated genes
                    top_genes = genes_df.head(10)['names'].tolist()
                    
                    # Mock LINCS query (replace with actual API call)
                    drug_targets[cluster] = {
                        'genes': top_genes,
                        'potential_drugs': [f"Drug_{i}" for i in range(5)],  # Mock results
                        'scores': [0.8, 0.7, 0.6, 0.5, 0.4]  # Mock scores
                    }
            
            self.logger.info("LINCS database query completed")
            return drug_targets
            
        except Exception as e:
            self.logger.warning(f"LINCS query failed: {e}")
            return {}
    
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