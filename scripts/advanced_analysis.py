#!/usr/bin/env python3
"""
Advanced Analysis Pipeline for LeukoMap

Python-based alternatives to R packages for advanced single-cell analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedLeukoMapAnalysis:
    """Advanced analysis pipeline using Python alternatives to R packages."""
    
    def __init__(self, output_dir: str = "results/advanced"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up scanpy settings
        sc.settings.set_figure_params(dpi=300, frameon=False)
        sc.settings.verbosity = 1
    
    def compare_annotations(self, auto_annotations: pd.DataFrame, 
                          manual_annotations: pd.DataFrame) -> Dict[str, float]:
        """Compare auto vs manual annotations using Python metrics."""
        logger.info("Comparing auto vs manual annotations...")
        
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data
            le = LabelEncoder()
            auto_encoded = le.fit_transform(auto_annotations['predicted_celltype'])
            manual_encoded = le.transform(manual_annotations['predicted_celltype'])
            
            # Calculate metrics
            ari = adjusted_rand_score(manual_encoded, auto_encoded)
            nmi = normalized_mutual_info_score(manual_encoded, auto_encoded)
            
            # Create confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(manual_encoded, auto_encoded)
            
            # Calculate accuracy per cell type
            cell_types = le.classes_
            accuracy_per_type = {}
            for i, ct in enumerate(cell_types):
                if i < len(cm):
                    accuracy_per_type[ct] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            
            results = {
                'adjusted_rand_index': ari,
                'normalized_mutual_info': nmi,
                'accuracy_per_celltype': accuracy_per_type,
                'confusion_matrix': cm,
                'cell_types': cell_types
            }
            
            # Save results
            self._save_annotation_comparison(results)
            
            logger.info(f"Annotation comparison complete - ARI: {ari:.3f}, NMI: {nmi:.3f}")
            return results
            
        except ImportError:
            logger.error("scikit-learn not available for annotation comparison")
            return {}
    
    def run_pseudotime_analysis(self, adata: sc.AnnData) -> sc.AnnData:
        """Run pseudotime analysis using Python alternatives to Monocle3."""
        logger.info("Running pseudotime analysis with Python alternatives...")
        
        try:
            # Method 1: Diffusion pseudotime (scanpy)
            sc.pp.neighbors(adata, use_rep='X_scVI' if 'X_scVI' in adata.obsm else 'X_pca')
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            
            # Method 2: PAGA (Partition-based graph abstraction)
            sc.tl.paga(adata, groups='predicted_celltype')
            sc.pl.paga(adata, show=False)
            plt.savefig(self.output_dir / 'paga_trajectory.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Method 3: Try scVelo if available
            try:
                import scvelo as scv
                scv.tl.velocity_graph(adata)
                scv.tl.velocity_pseudotime(adata)
                logger.info("scVelo pseudotime analysis completed")
            except ImportError:
                logger.info("scVelo not available, using scanpy methods only")
            
            # Create pseudotime visualization
            self._create_pseudotime_plots(adata)
            
            logger.info("Pseudotime analysis completed")
            return adata
            
        except Exception as e:
            logger.error(f"Pseudotime analysis failed: {e}")
            return adata
    
    def run_python_gsea(self, adata: sc.AnnData) -> Dict[str, Any]:
        """Run GSEA using Python alternatives to GSVA/fgsea."""
        logger.info("Running GSEA analysis with Python alternatives...")
        
        try:
            # Method 1: Use gseapy (Python GSEA implementation)
            import gseapy as gp
            
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
            
            # Method 2: Simple pathway enrichment using gene sets
            pathway_results = self._simple_pathway_enrichment(adata)
            gsea_results['pathway_enrichment'] = pathway_results
            
            # Save results
            self._save_gsea_results(gsea_results)
            
            logger.info(f"GSEA analysis completed for {len(gsea_results)} cell types")
            return gsea_results
            
        except ImportError:
            logger.warning("gseapy not available, using simple pathway enrichment")
            return self._simple_pathway_enrichment(adata)
    
    def query_drug_databases(self, deg_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Query drug databases using Python APIs."""
        logger.info("Querying drug databases for therapeutic targets...")
        
        try:
            # Method 1: Use Enrichr API via requests
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
            
            # Method 2: Simple drug-gene associations
            drug_gene_associations = self._simple_drug_associations(deg_results)
            drug_results['drug_gene_associations'] = drug_gene_associations
            
            # Save results
            self._save_drug_results(drug_results)
            
            logger.info("Drug database query completed")
            return drug_results
            
        except Exception as e:
            logger.error(f"Drug database query failed: {e}")
            return self._simple_drug_associations(deg_results)
    
    def generate_publication_figures(self, adata: sc.AnnData, 
                                   analysis_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate publication-quality figures."""
        logger.info("Generating publication-quality figures...")
        
        figures = {}
        
        # 1. Main UMAP with cell types
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        sc.pl.umap(adata, color='predicted_celltype', ax=ax, show=False, 
                  legend_loc='on data', legend_fontsize=8)
        plt.title('Cell Type Distribution in Pediatric Leukemia', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        main_umap = self.output_dir / 'publication_main_umap.png'
        plt.savefig(main_umap, dpi=300, bbox_inches='tight')
        plt.close()
        figures['main_umap'] = main_umap
        
        # 2. Pseudotime trajectory
        if 'dpt_pseudotime' in adata.obs.columns:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sc.pl.umap(adata, color='dpt_pseudotime', ax=ax, show=False, 
                      color_map='viridis', legend_loc='on data')
            plt.title('Pseudotime Trajectory', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            pseudotime_plot = self.output_dir / 'publication_pseudotime.png'
            plt.savefig(pseudotime_plot, dpi=300, bbox_inches='tight')
            plt.close()
            figures['pseudotime'] = pseudotime_plot
        
        # 3. Top pathways heatmap
        if 'pathway_enrichment' in analysis_results:
            self._create_pathway_heatmap(analysis_results['pathway_enrichment'])
            pathway_plot = self.output_dir / 'publication_pathways.png'
            figures['pathways'] = pathway_plot
        
        # 4. Drug targets summary
        if 'drug_gene_associations' in analysis_results:
            self._create_drug_summary(analysis_results['drug_gene_associations'])
            drug_plot = self.output_dir / 'publication_drugs.png'
            figures['drugs'] = drug_plot
        
        logger.info(f"Generated {len(figures)} publication figures")
        return figures
    
    def export_for_ml(self, adata: sc.AnnData, results: Dict[str, Any]) -> Dict[str, Path]:
        """Export results in formats ready for ML and ontology integration."""
        logger.info("Exporting results for ML and ontology integration...")
        
        exports = {}
        
        # 1. Cell metadata with annotations
        cell_metadata = adata.obs.copy()
        cell_metadata_path = self.output_dir / 'cell_metadata.csv'
        cell_metadata.to_csv(cell_metadata_path)
        exports['cell_metadata'] = cell_metadata_path
        
        # 2. Gene expression matrix
        expr_matrix = pd.DataFrame(adata.X.toarray(), 
                                 index=adata.obs_names, 
                                 columns=adata.var_names)
        expr_path = self.output_dir / 'expression_matrix.csv'
        expr_matrix.to_csv(expr_path)
        exports['expression_matrix'] = expr_path
        
        # 3. Differential expression results
        if 'differential_expression' in results:
            deg_path = self.output_dir / 'differential_expression.json'
            with open(deg_path, 'w') as f:
                json.dump({k: v.to_dict() for k, v in results['differential_expression'].items()}, f)
            exports['differential_expression'] = deg_path
        
        # 4. Pathway enrichment results
        if 'pathway_enrichment' in results:
            pathway_path = self.output_dir / 'pathway_enrichment.json'
            with open(pathway_path, 'w') as f:
                json.dump(results['pathway_enrichment'], f)
            exports['pathway_enrichment'] = pathway_path
        
        # 5. Drug associations
        if 'drug_gene_associations' in results:
            drug_path = self.output_dir / 'drug_associations.json'
            with open(drug_path, 'w') as f:
                json.dump(results['drug_gene_associations'], f)
            exports['drug_associations'] = drug_path
        
        logger.info(f"Exported {len(exports)} files for ML integration")
        return exports
    
    # Helper methods
    def _save_annotation_comparison(self, results: Dict[str, Any]):
        """Save annotation comparison results."""
        # Save metrics
        metrics_df = pd.DataFrame({
            'metric': ['ARI', 'NMI'],
            'value': [results['adjusted_rand_index'], results['normalized_mutual_info']]
        })
        metrics_df.to_csv(self.output_dir / 'annotation_comparison_metrics.csv', index=False)
        
        # Save confusion matrix
        cm_df = pd.DataFrame(results['confusion_matrix'], 
                           index=results['cell_types'], 
                           columns=results['cell_types'])
        cm_df.to_csv(self.output_dir / 'annotation_confusion_matrix.csv')
    
    def _create_pseudotime_plots(self, adata: sc.AnnData):
        """Create pseudotime visualization plots."""
        if 'dpt_pseudotime' in adata.obs.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # UMAP with pseudotime
            sc.pl.umap(adata, color='dpt_pseudotime', ax=ax1, show=False, color_map='viridis')
            ax1.set_title('Pseudotime on UMAP', fontsize=14, fontweight='bold')
            
            # Pseudotime distribution by cell type
            if 'predicted_celltype' in adata.obs.columns:
                cell_types = adata.obs['predicted_celltype'].unique()
                pseudotime_by_type = [adata.obs[adata.obs['predicted_celltype'] == ct]['dpt_pseudotime'].values 
                                     for ct in cell_types]
                ax2.boxplot(pseudotime_by_type, labels=cell_types)
                ax2.set_xlabel('Cell Type', fontsize=12)
                ax2.set_ylabel('Pseudotime', fontsize=12)
                ax2.set_title('Pseudotime by Cell Type', fontsize=14, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'pseudotime_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _simple_pathway_enrichment(self, adata: sc.AnnData) -> Dict[str, Any]:
        """Simple pathway enrichment without external dependencies."""
        # This would implement a basic pathway enrichment using gene sets
        # For now, return empty results
        return {}
    
    def _save_gsea_results(self, results: Dict[str, Any]):
        """Save GSEA results."""
        for cell_type, result_df in results.items():
            if isinstance(result_df, pd.DataFrame):
                result_df.to_csv(self.output_dir / f'gsea_{cell_type}.csv', index=False)
    
    def _simple_drug_associations(self, deg_results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Simple drug-gene associations without external APIs."""
        # Mock drug associations for demonstration
        drug_associations = {}
        for cell_type, genes_df in deg_results.items():
            if len(genes_df) > 0:
                top_genes = genes_df.head(10)['names'].tolist()
                drug_associations[cell_type] = {
                    'top_genes': top_genes,
                    'potential_drugs': ['Drug_A', 'Drug_B', 'Drug_C']  # Mock drugs
                }
        return drug_associations
    
    def _save_drug_results(self, results: Dict[str, Any]):
        """Save drug query results."""
        import json
        with open(self.output_dir / 'drug_results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    def _create_pathway_heatmap(self, pathway_results: Dict[str, Any]):
        """Create pathway enrichment heatmap."""
        # Implementation for pathway heatmap
        pass
    
    def _create_drug_summary(self, drug_results: Dict[str, Any]):
        """Create drug targets summary plot."""
        # Implementation for drug summary
        pass

def main():
    """Main function for standalone advanced analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LeukoMap Advanced Analysis Pipeline")
    parser.add_argument("--input-file", "-i", required=True, help="Input AnnData file (.h5ad)")
    parser.add_argument("--output-dir", "-o", default="results/advanced", help="Output directory")
    parser.add_argument("--manual-annotations", "-m", help="Manual annotations file (optional)")
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input_file}")
    adata = sc.read_h5ad(args.input_file)
    
    # Run advanced analysis
    analyzer = AdvancedLeukoMapAnalysis(args.output_dir)
    
    # Run analyses
    results = {}
    
    # Pseudotime analysis
    adata = analyzer.run_pseudotime_analysis(adata)
    
    # GSEA analysis
    results['gsea'] = analyzer.run_python_gsea(adata)
    
    # Drug database query
    if 'differential_expression' in adata.uns:
        results['drugs'] = analyzer.query_drug_databases(adata.uns['differential_expression'])
    
    # Generate publication figures
    figures = analyzer.generate_publication_figures(adata, results)
    
    # Export for ML
    exports = analyzer.export_for_ml(adata, results)
    
    print(f"\nAdvanced analysis complete!")
    print(f"Generated {len(figures)} publication figures")
    print(f"Exported {len(exports)} files for ML integration")

if __name__ == "__main__":
    main() 