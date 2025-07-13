#!/usr/bin/env python3
"""
Basic LeukoMap Analysis Example.

This example demonstrates the new modular architecture of LeukoMap,
showing how to use the object-oriented classes for data loading,
preprocessing, and analysis.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import leukomap
from leukomap import (
    AnalysisConfig,
    LeukoMapAnalysis,
    DataManager,
    PreprocessingManager
)


def main():
    """Run a basic LeukoMap analysis using the new architecture."""
    
    # 1. Create configuration
    config = AnalysisConfig(
        data_path=Path("data"),  # Path to your data
        output_dir=Path("results"),
        min_genes=200,
        min_cells=3,
        n_latent=10,
        max_epochs=100  # Reduced for example
    )
    
    print("Configuration created:")
    print(f"  Data path: {config.data_path}")
    print(f"  Output directory: {config.output_dir}")
    print(f"  Min genes: {config.min_genes}")
    print(f"  Min cells: {config.min_cells}")
    print()
    
    # 2. Create data manager and load data
    print("Loading and validating data...")
    data_manager = DataManager(config)
    
    try:
        # Load and validate data
        adata = data_manager.load_and_validate()
        print(f"Data loaded successfully: {adata.n_obs} cells, {adata.n_vars} genes")
        
        # Save loaded data
        data_path = data_manager.save_data(adata, "raw_data.h5ad")
        print(f"Raw data saved to: {data_path}")
        
    except FileNotFoundError:
        print("Data not found. Creating example with existing integrated data...")
        # Try to load existing integrated data
        import scanpy as sc
        adata = sc.read_h5ad("results/adata_integrated_healthy_quick.h5ad")
        print(f"Loaded existing data: {adata.n_obs} cells, {adata.n_vars} genes")
    
    print()
    
    # 3. Create preprocessing manager and preprocess data
    print("Preprocessing data...")
    preprocessing_manager = PreprocessingManager(config)
    
    # Preprocess data
    preprocessed_adata = preprocessing_manager.preprocess_data(adata, save_results=True)
    print(f"Preprocessing complete: {preprocessed_adata.n_obs} cells, {preprocessed_adata.n_vars} genes")
    
    # Generate preprocessing report
    report_path = preprocessing_manager.generate_preprocessing_report(adata, preprocessed_adata)
    print(f"Preprocessing report saved to: {report_path}")
    
    print()
    
    # 4. Create main analysis object
    print("Setting up main analysis...")
    analysis = LeukoMapAnalysis(config)
    
    # Add processors to pipeline (example)
    # Note: In a real analysis, you would add your specific processors here
    print("Analysis object created successfully")
    print("  - Pipeline ready for processors")
    print("  - Result tracker initialized")
    print("  - Report generation available")
    
    print()
    
    # 5. Demonstrate result tracking
    print("Demonstrating result tracking...")
    
    # Store some results
    analysis.results.store("raw_data", adata, {"stage": "data_loading"})
    analysis.results.store("preprocessed_data", preprocessed_adata, {"stage": "preprocessing"})
    
    # List stored results
    print("Stored results:")
    for key in analysis.results.list_results():
        metadata = analysis.results.get_metadata(key)
        print(f"  - {key}: {metadata}")
    
    # Save results to disk
    for key in analysis.results.list_results():
        filepath = analysis.results.save_to_disk(key)
        print(f"  Saved {key} to: {filepath}")
    
    print()
    
    # 6. Generate analysis report
    print("Generating analysis report...")
    report_path = analysis.save_analysis_report()
    print(f"Analysis report saved to: {report_path}")
    
    print()
    print("=" * 60)
    print("BASIC ANALYSIS COMPLETE")
    print("=" * 60)
    print()
    print("This example demonstrates:")
    print("  ✓ Configuration management")
    print("  ✓ Data loading and validation")
    print("  ✓ Preprocessing pipeline")
    print("  ✓ Result tracking and storage")
    print("  ✓ Report generation")
    print()
    print("Next steps:")
    print("  - Add specific processors to the pipeline")
    print("  - Implement training and analysis modules")
    print("  - Add visualization and annotation capabilities")
    print()
    print("The new architecture provides:")
    print("  - Modularity: Each component is independent")
    print("  - Composability: Easy to combine components")
    print("  - Extensibility: Simple to add new functionality")
    print("  - Maintainability: Clear separation of concerns")


if __name__ == "__main__":
    main() 