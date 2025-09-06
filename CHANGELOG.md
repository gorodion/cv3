# Changelog
## [1.3.1] - 2025-09-07

### Added
* default line_type value moved to cv3.opt
* updated threshold function with type parameter and relative values support
* new test_threshold_all_types in test_processing.py
* opt functions validation and tests in test_other
* examples in video.py for VideoCapture, VideoWriter, and Video functions

### Changed
* got rid of f-strings for compatibility with early python versions
* got rid of TODOs
* small docs corrections

### Deprecated

### Removed

### Fixed

## [1.3.0] - 2025-08-31

### Added
* Comprehensive documentation system with Sphinx
* Detailed API reference for all modules
* Introduction and features documentation
* Practical examples for common use cases
* Module-level documentation for all cv3 modules
* Function-level documentation with parameters, returns, and examples
* New drawing functions: arrowedLine (as arrow), ellipse, drawMarker (as marker), getTextSize
* EXPERIMENTAL flag in opt.py for new features
* Fill parameter for rectangle, rectangles, and circle functions
* Complete test suite for new drawing functions
* Restructured draw module with internal functions moved to _draw.py
* New transform module with internal functions in _transform.py
* Module-level documentation for all cv3 modules
* Read the Docs configuration files (readthedocs.yaml and docs/requirements.txt)
* Core concepts documentation page with detailed package features
* Project URLs in setup.py for documentation and source code

### Changed
* Improved documentation for all existing functions
* Restructured draw module implementation
* Moved internal transform functions to private module
* Enhanced docstrings with detailed parameter descriptions
* Updated project version to 1.3.0
* Updated installation instructions in documentation and README
* Updated README.md to match documentation content

### Deprecated

### Removed

### Fixed
* Documentation inconsistencies
* Module organization issues
* Error messages in draw.py experimental functions to use opt.set_exp() instead of "Set opt.EXPERIMENTAL = True"

## [1.2.1] - 2023-06-25

### Added
* support read and write image using non-ascii paths. Parameter `ascii=True` in cv3.imwrite
* cv3.imdecode
### Changed
### Deprecated
* flag='alpha' in cv3.imread. Use 'unchanged' instead
### Removed
### Fixed