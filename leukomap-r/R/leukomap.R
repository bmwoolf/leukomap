#' Initialize LeukoMap Python interface
#' 
#' Sets up the Python environment and imports the LeukoMap package.
#' 
#' @param python_path Optional path to Python virtual environment
#' @return LeukoMap Python module
#' @export
#' @examples
#' \dontrun{
#' leukomap <- init_leukomap()
#' }
init_leukomap <- function(python_path = NULL) {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    stop("reticulate package is required. Install with: install.packages('reticulate')")
  }
  
  # Use specified Python path or default
  if (!is.null(python_path)) {
    reticulate::use_python(python_path)
  } else {
    # Try to use current Python environment
    tryCatch({
      reticulate::use_python(reticulate::py_config()$python)
    }, error = function(e) {
      message("Using default Python environment")
    })
  }
  
  # Import LeukoMap
  tryCatch({
    leukomap <- reticulate::import("leukomap")
    message("LeukoMap Python package loaded successfully")
    return(leukomap)
  }, error = function(e) {
    stop("Failed to import LeukoMap Python package. Error: ", e$message)
  })
}

#' Analyze leukemia single-cell data
#' 
#' Main analysis function that runs the complete LeukoMap pipeline.
#' 
#' @param data_path Path to input data file (H5AD format)
#' @param output_dir Directory to save results (default: "results")
#' @param python_path Optional path to Python virtual environment
#' @return Analysis results
#' @export
#' @examples
#' \dontrun{
#' results <- analyze_leukemia("data.h5ad", output_dir = "my_results")
#' }
analyze_leukemia <- function(data_path, output_dir = "results", python_path = NULL) {
  # Validate input
  if (!file.exists(data_path)) {
    stop("Data file not found: ", data_path)
  }
  
  # Initialize LeukoMap
  leukomap <- init_leukomap(python_path)
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Run analysis
  tryCatch({
    message("Starting LeukoMap analysis...")
    results <- leukomap$analyze(data_path, output_dir = output_dir)
    message("Analysis completed successfully")
    return(results)
  }, error = function(e) {
    stop("Analysis failed: ", e$message)
  })
}

#' Load and preprocess data
#' 
#' Load single-cell data and run preprocessing pipeline.
#' 
#' @param data_path Path to input data file
#' @param python_path Optional path to Python virtual environment
#' @return Preprocessed AnnData object
#' @export
load_and_preprocess <- function(data_path, python_path = NULL) {
  leukomap <- init_leukomap(python_path)
  
  tryCatch({
    message("Loading and preprocessing data...")
    adata <- leukomap$data_loading$load_data(data_path)
    preprocessed <- leukomap$preprocessing$preprocess_data(adata)
    message("Data preprocessing completed")
    return(preprocessed)
  }, error = function(e) {
    stop("Data preprocessing failed: ", e$message)
  })
}

#' Annotate cell types
#' 
#' Run cell type annotation on preprocessed data.
#' 
#' @param adata Preprocessed AnnData object
#' @param method Annotation method ("celltypist", "azimuth", "singler")
#' @param python_path Optional path to Python virtual environment
#' @return Annotated AnnData object
#' @export
annotate_cell_types <- function(adata, method = "celltypist", python_path = NULL) {
  leukomap <- init_leukomap(python_path)
  
  tryCatch({
    message("Running cell type annotation with method: ", method)
    annotator <- leukomap$cell_type_annotation$CellTypeAnnotator()
    annotated <- annotator$annotate(adata, method = method)
    message("Cell type annotation completed")
    return(annotated)
  }, error = function(e) {
    stop("Cell type annotation failed: ", e$message)
  })
}

#' Get package information
#' 
#' Display information about the LeukoMap package and Python environment.
#' 
#' @param python_path Optional path to Python virtual environment
#' @return Package information
#' @export
package_info <- function(python_path = NULL) {
  leukomap <- init_leukomap(python_path)
  
  info <- list(
    python_version = reticulate::py_config()$version,
    leukomap_version = tryCatch(leukomap$`__version__`, error = function(e) "Unknown"),
    available_methods = c("celltypist", "azimuth", "singler"),
    reticulate_available = TRUE
  )
  
  return(info)
} 