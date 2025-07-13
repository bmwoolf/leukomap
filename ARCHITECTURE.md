# LeukoMap Architecture

## Overview

LeukoMap has been redesigned as a modern, modular software package with emphasis on **modularity**, **composability**, and **extensibility**. The new architecture follows object-oriented design principles and provides a clean separation of concerns.

## Core Architecture

### 1. Configuration Management

The `AnalysisConfig` class provides centralized configuration management:

```python
from leukomap import AnalysisConfig

config = AnalysisConfig(
    data_path=Path("data"),
    output_dir=Path("results"),
    min_genes=200,
    min_cells=3,
    n_latent=10,
    max_epochs=400
)
```

**Benefits:**
- Centralized parameter management
- Type safety with dataclass
- Automatic directory creation
- Easy configuration sharing

### 2. Pipeline Architecture

The `Pipeline` class orchestrates analysis steps:

```python
from leukomap import Pipeline, DataLoader, QualityControl

pipeline = Pipeline(config)
pipeline.add_processor(DataLoader(config))
pipeline.add_processor(QualityControl(config))
results = pipeline.run()
```

**Benefits:**
- Sequential execution with error handling
- Result tracking at each stage
- Easy to add/remove steps
- Automatic result saving

### 3. Processor Interface

All analysis components implement the `BaseProcessor` interface:

```python
class MyProcessor(BaseProcessor):
    def process(self, data: Any) -> Any:
        # Process data
        return processed_data
    
    def save_results(self, data: Any, output_path: Path) -> None:
        # Save results
        pass
```

**Benefits:**
- Consistent interface across components
- Easy to extend with new processors
- Built-in validation and error handling
- Automatic stage tracking

## Module Structure

### Core Module (`leukomap.core`)

**Classes:**
- `AnalysisConfig`: Configuration management
- `AnalysisStage`: Enumeration of analysis stages
- `BaseProcessor`: Abstract base class for processors
- `DataProcessor`: Base class for data processors
- `Pipeline`: Main pipeline orchestrator
- `ResultTracker`: Result management and storage
- `LeukoMapAnalysis`: High-level analysis interface

**Purpose:** Provides the foundational architecture and interfaces.

### Data Module (`leukomap.data`)

**Classes:**
- `DataLoader`: Load data from various formats
- `DataValidator`: Validate data quality and integrity
- `DataManager`: High-level data management interface

**Purpose:** Handles data loading, validation, and management.

### Preprocessing Module (`leukomap.preprocessing`)

**Classes:**
- `QualityControl`: Filter cells and genes
- `Normalizer`: Normalize data (library size, log transform)
- `FeatureSelector`: Select highly variable genes
- `Scaler`: Scale data for analysis
- `PreprocessingPipeline`: Complete preprocessing workflow
- `PreprocessingManager`: High-level preprocessing interface

**Purpose:** Handles data preprocessing and quality control.

## Usage Examples

### Basic Analysis

```python
from leukomap import LeukoMapAnalysis, AnalysisConfig

# Create configuration
config = AnalysisConfig(
    data_path=Path("data"),
    output_dir=Path("results")
)

# Create analysis object
analysis = LeukoMapAnalysis(config)

# Run full analysis
results = analysis.run_full_analysis()
```

### Modular Usage

```python
from leukomap import DataManager, PreprocessingManager

# Load and validate data
data_manager = DataManager(config)
adata = data_manager.load_and_validate()

# Preprocess data
preprocessing_manager = PreprocessingManager(config)
preprocessed_adata = preprocessing_manager.preprocess_data(adata)
```

### Custom Pipeline

```python
from leukomap import Pipeline, DataLoader, QualityControl

# Create custom pipeline
pipeline = Pipeline(config)
pipeline.add_processor(DataLoader(config))
pipeline.add_processor(QualityControl(config))

# Run pipeline
results = pipeline.run()
```

## Key Design Principles

### 1. Modularity

Each component is independent and can be used separately:

```python
# Use only data loading
from leukomap import DataManager
data_manager = DataManager(config)
adata = data_manager.load_and_validate()

# Use only preprocessing
from leukomap import PreprocessingManager
preprocessing_manager = PreprocessingManager(config)
preprocessed_adata = preprocessing_manager.preprocess_data(adata)
```

### 2. Composability

Components can be easily combined:

```python
# Combine data loading and preprocessing
data_manager = DataManager(config)
preprocessing_manager = PreprocessingManager(config)

adata = data_manager.load_and_validate()
preprocessed_adata = preprocessing_manager.preprocess_data(adata)
```

### 3. Extensibility

Easy to add new functionality:

```python
class MyCustomProcessor(BaseProcessor):
    def process(self, data: Any) -> Any:
        # Custom processing logic
        return processed_data
    
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.ANALYSIS

# Add to pipeline
pipeline.add_processor(MyCustomProcessor(config))
```

### 4. Configuration-Driven

All behavior is controlled through configuration:

```python
config = AnalysisConfig(
    min_genes=200,      # Quality control
    min_cells=3,        # Quality control
    n_latent=10,        # Training
    max_epochs=400,     # Training
    resolution=0.5      # Clustering
)
```

## Result Management

### Result Tracking

The `ResultTracker` class provides centralized result management:

```python
# Store results
analysis.results.store("raw_data", adata, {"stage": "data_loading"})
analysis.results.store("preprocessed_data", preprocessed_adata, {"stage": "preprocessing"})

# Retrieve results
raw_data = analysis.results.get("raw_data")
metadata = analysis.results.get_metadata("raw_data")

# List all results
for key in analysis.results.list_results():
    print(f"Available result: {key}")
```

### Automatic Saving

Results are automatically saved to disk:

```python
# Save to disk
filepath = analysis.results.save_to_disk("raw_data", "my_data.h5ad")
```

## Error Handling

### Graceful Degradation

Components handle errors gracefully:

```python
try:
    adata = data_manager.load_and_validate()
except FileNotFoundError:
    print("Data not found, using fallback")
    adata = load_fallback_data()
```

### Validation

Built-in validation at each step:

```python
class MyProcessor(BaseProcessor):
    def validate_input(self, data: Any) -> bool:
        if not isinstance(data, ad.AnnData):
            self.logger.error("Input must be AnnData")
            return False
        return True
```

## Logging and Reporting

### Comprehensive Logging

All components provide detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# All components log their progress
data_manager = DataManager(config)
adata = data_manager.load_and_validate()
# Output: INFO - Loading data from: data/
# Output: INFO - Data validation passed
```

### Automatic Report Generation

Reports are generated automatically:

```python
# Generate preprocessing report
report_path = preprocessing_manager.generate_preprocessing_report(
    original_adata, preprocessed_adata
)

# Generate analysis report
report_path = analysis.save_analysis_report()
```

## Migration from Legacy Code

### Backward Compatibility

Legacy functions are still available:

```python
# Old way (still works)
from leukomap import load_data, preprocess
adata = load_data("data")
adata = preprocess(adata)

# New way (recommended)
from leukomap import DataManager, PreprocessingManager
data_manager = DataManager(config)
preprocessing_manager = PreprocessingManager(config)

adata = data_manager.load_and_validate()
preprocessed_adata = preprocessing_manager.preprocess_data(adata)
```

### Gradual Migration

You can migrate gradually:

```python
# Start with new data loading
from leukomap import DataManager
data_manager = DataManager(config)
adata = data_manager.load_and_validate()

# Use legacy preprocessing for now
from leukomap import preprocess
adata = preprocess(adata)

# Later migrate to new preprocessing
from leukomap import PreprocessingManager
preprocessing_manager = PreprocessingManager(config)
adata = preprocessing_manager.preprocess_data(adata)
```

## Future Extensions

### Adding New Processors

```python
class MyAnalysisProcessor(BaseProcessor):
    def process(self, data: ad.AnnData) -> ad.AnnData:
        # Perform analysis
        return analyzed_data
    
    def get_stage(self) -> AnalysisStage:
        return AnalysisStage.ANALYSIS

# Use in pipeline
pipeline.add_processor(MyAnalysisProcessor(config))
```

### Adding New Data Formats

```python
class MyDataLoader(DataLoader):
    def _load_my_format(self, data_dir: Path) -> ad.AnnData:
        # Load custom format
        return adata
    
    def _is_my_format(self, data_dir: Path) -> bool:
        # Check if data is in custom format
        return True
```

### Adding New Configuration Options

```python
@dataclass
class ExtendedConfig(AnalysisConfig):
    my_parameter: str = "default_value"
    my_other_parameter: int = 42
```

## Benefits of the New Architecture

### For Developers

1. **Maintainability**: Clear separation of concerns
2. **Testability**: Each component can be tested independently
3. **Extensibility**: Easy to add new functionality
4. **Reusability**: Components can be reused across projects

### For Users

1. **Flexibility**: Use only what you need
2. **Clarity**: Clear, intuitive interfaces
3. **Reliability**: Built-in error handling and validation
4. **Documentation**: Comprehensive logging and reporting

### For the Project

1. **Scalability**: Easy to add new features
2. **Collaboration**: Clear interfaces for contributors
3. **Quality**: Built-in validation and testing
4. **Documentation**: Self-documenting code structure

## Conclusion

The new LeukoMap architecture provides a solid foundation for modern single-cell RNA-seq analysis. It combines the power of object-oriented design with the flexibility of modular components, making it easy to use, extend, and maintain.

Whether you're a beginner looking for simple analysis workflows or an advanced user building custom pipelines, the new architecture provides the tools you need to succeed. 