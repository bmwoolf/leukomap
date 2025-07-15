# LeukoMap R Package

R interface for the LeukoMap Python package for leukemia single-cell RNA-seq analysis.

## Installation

### Prerequisites

1. **R** (version 4.0 or higher)
2. **Python** with LeukoMap package installed
3. **reticulate** R package

### Install R Dependencies

```r
# Install reticulate if not already installed
install.packages("reticulate")

# Install LeukoMap R package (from source)
install.packages("leukomap-r", repos = NULL, type = "source")
```

## Usage

### Basic Analysis

```r
library(leukomap)

# Run complete analysis pipeline
results <- analyze_leukemia("data.h5ad", output_dir = "results")
```

### Step-by-Step Analysis

```r
library(leukomap)

# Load and preprocess data
adata <- load_and_preprocess("data.h5ad")

# Annotate cell types
annotated <- annotate_cell_types(adata, method = "celltypist")

# Get package information
info <- package_info()
print(info)
```

### Specify Python Environment

```r
# Use specific Python virtual environment
results <- analyze_leukemia("data.h5ad", python_path = "/path/to/venv/bin/python")
```

## Functions

- `init_leukomap()`: Initialize Python interface
- `analyze_leukemia()`: Run complete analysis pipeline
- `load_and_preprocess()`: Load and preprocess data
- `annotate_cell_types()`: Annotate cell types
- `package_info()`: Get package information

## Examples

See the `examples/` directory for detailed usage examples.

## Troubleshooting

### Python Environment Issues

If you encounter Python environment issues:

1. Ensure LeukoMap Python package is installed
2. Specify the correct Python path:
   ```r
   results <- analyze_leukemia("data.h5ad", python_path = "/path/to/python")
   ```

### Reticulate Issues

If reticulate fails to import LeukoMap:

```r
# Check Python configuration
reticulate::py_config()

# Reinstall reticulate if needed
install.packages("reticulate")
```

## License

MIT License 