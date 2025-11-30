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
python map_chemicals.py --folder /path/to/data/folder
```

For agilent data, add the --Agilent flag to the command above, i.e.
```bash
python map_chemicals.py --folder /path/to/data/folder --Agilent
```
