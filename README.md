# SpecMatch Application

SpecMatch is a PyQt5-based desktop application for spectral image analysis and material classification. It uses machine learning models (ONNX format) to classify materials in hyperspectral images and provides interactive visualization and analysis tools.

## Features

- **Spectral Image Analysis**: Load and analyze hyperspectral measurements in ENVI format (.dat/.hdr files)
- **Material Classification**: Classify materials using pre-trained ONNX models
- **Interactive Visualization**: 
  - View spectral images with zoom/pan capabilities
  - Display pixel-level spectra and classification probabilities
  - Toggle between raw, denoised, and grouped classification views
- **Image Processing**:
  - Automatic denoising with morphological operations
  - Particle detection and grouping
  - Contour-based segmentation
- **Batch Processing**: Process multiple measurements in queue
- **Data Export**: Export particle and pixel data to CSV files with probability distributions

## Prerequisites

- Python 3.13 or higher
- pip (Python package installer)

## Installation

### Step 1: Clone or Download the Project

If you haven't already, download or clone this project to your local machine.

### Step 2: Create a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
# Navigate to the project directory
cd /path/to/SpecMatch-App-main

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required libraries using the requirements.txt file:

```bash
pip install -r requirements.txt
```

This will install the following packages:
- **PyQt5**: GUI framework
- **opencv-python**: Image processing
- **numpy**: Numerical computations
- **pyyaml**: YAML file parsing
- **matplotlib**: Plotting and visualization
- **pandas**: Data manipulation
- **onnxruntime**: ONNX model inference
- **scikit-learn**: Machine learning utilities (PCA)
- **spectral**: Hyperspectral image processing

## Project Structure

```
SpecMatch-App-main/
├── main.py              # Main application file
├── main.ui              # PyQt5 UI definition file
├── utils.py             # Utility functions (ENVI reader, color conversion)
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── SpecMatch.spec      # PyInstaller specification (for building executables)
├── models/             # Directory for ONNX models (not included)
│   └── m1/
│       ├── model.onnx
│       └── metadata.yaml
└── debugging/          # Directory for debug output images (created automatically)
```

## Required Model Files

The application requires a trained model in ONNX format. The model should be placed in a directory (e.g., `models/m1/`) with the following structure:

```
models/m1/
├── model.onnx          # ONNX model file
└── metadata.yaml       # Model metadata including substance colors and mappings
```

The `metadata.yaml` file should contain:
- `substance_colors`: Dictionary mapping substance names to hex colors
- `substance_to_idx`: Dictionary mapping substance names to indices
- `idx_to_substance`: Dictionary mapping indices to substance names

## Usage

### Running the Application

Once dependencies are installed, run the application with:

```bash
python main.py
```

### Basic Workflow

1. **Select Model**: Click "Select Model" and choose your model folder (e.g., `models/m1/`)
2. **Select Measurement**: Click "Select Measurement" and choose a folder containing ENVI format hyperspectral data
   - The folder should contain files named `{foldername}.dat` and `{foldername}.hdr`, or `stitched_{foldername}.dat` and `stitched_{foldername}.hdr`
3. **Run Analysis**: Click "Run" to perform spectral classification
4. **View Results**: 
   - Use the dropdown to toggle between [1] Raw, [2] Denoised, and [3] Grouped views
   - Hover over pixels to see their spectrum and classification probabilities
   - Click on pixels to save their spectrum as PNG
5. **Export Data**: Click "Export" to save particle and pixel data to CSV files

### Batch Processing

1. Click "Add Measurement" to add individual measurement folders to the queue
2. Or click "Add Parent" to add all valid measurement subfolders from a parent directory
3. Configure options (e.g., "Skip existing" to avoid reprocessing)
4. Click "Run Queue" to process all measurements automatically
5. View processed measurements in the "Processed" list

### Keyboard Shortcuts

- **1**: Switch to Raw view
- **2**: Switch to Denoised view
- **3**: Switch to Grouped view

## Input Data Format

The application expects hyperspectral measurements in ENVI format:
- `.dat` file: Binary data file containing spectral information
- `.hdr` file: Header file with metadata (wavelengths, dimensions, etc.)

## Output Files

When exporting, the application creates:
- `particledata_SpecMatch.csv`: Particle-level data with features and probabilities
- `pixeldata_SpecMatch.csv`: Pixel-level data with individual probabilities
- `specmatch_session.pkl`: Cached analysis session for faster reloading

## Development Mode

Set `DEV_MODE = True` in `main.py` (line 33) to enable automatic loading of example data on startup.

## Troubleshooting

### Common Issues

1. **"No module named 'PyQt5'"**: Make sure you've installed all dependencies with `pip install -r requirements.txt`

2. **Model not loading**: Ensure your model folder contains both `model.onnx` and `metadata.yaml` files

3. **Measurement not loading**: Verify that your measurement folder contains properly named `.dat` and `.hdr` files

4. **Display issues on Linux**: The application uses PyQt5 with matplotlib. Ensure you have proper display drivers installed.

## Building Executable (Optional)

To create a standalone executable using PyInstaller:

```bash
pip install pyinstaller
pyinstaller SpecMatch.spec
```

The executable will be created in the `dist/` directory.

## License

[Add your license information here]

## Contact

[Add contact information here]
