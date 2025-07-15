#!/usr/bin/env Rscript
# Basic usage example for LeukoMap R package

# Load the package
library(leukomap)

# Example 1: Get package information
cat("=== LeukoMap Package Information ===\n")
info <- package_info()
print(info)

# Example 2: Initialize LeukoMap (without running analysis)
cat("\n=== Initializing LeukoMap ===\n")
leukomap <- init_leukomap()
cat("LeukoMap Python interface initialized successfully\n")

# Example 3: Check if data file exists and run analysis
data_file <- "cache/adata_preprocessed.h5ad"

if (file.exists(data_file)) {
  cat("\n=== Running Complete Analysis ===\n")
  cat("Using data file:", data_file, "\n")
  
  # Run complete analysis
  results <- analyze_leukemia(
    data_path = data_file,
    output_dir = "results/r_analysis"
  )
  
  cat("Analysis completed successfully!\n")
  cat("Results saved to: results/r_analysis\n")
  
} else {
  cat("\n=== Data File Not Found ===\n")
  cat("Data file not found:", data_file, "\n")
  cat("Please ensure the data file exists or update the path.\n")
  
  # Show available data files
  cat("\nAvailable data files:\n")
  if (dir.exists("cache")) {
    cache_files <- list.files("cache", pattern = "\\.h5ad$")
    if (length(cache_files) > 0) {
      for (file in cache_files) {
        cat("  cache/", file, "\n", sep = "")
      }
    } else {
      cat("  No H5AD files found in cache directory\n")
    }
  } else {
    cat("  Cache directory not found\n")
  }
}

# Example 4: Step-by-step analysis (if data available)
if (file.exists(data_file)) {
  cat("\n=== Step-by-Step Analysis ===\n")
  
  # Load and preprocess
  cat("Loading and preprocessing data...\n")
  adata <- load_and_preprocess(data_file)
  cat("Data loaded successfully\n")
  
  # Annotate cell types
  cat("Annotating cell types...\n")
  annotated <- annotate_cell_types(adata, method = "celltypist")
  cat("Cell type annotation completed\n")
  
  cat("Step-by-step analysis completed successfully!\n")
}

cat("\n=== Example completed ===\n") 