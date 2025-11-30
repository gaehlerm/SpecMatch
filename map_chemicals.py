#!/usr/bin/env python3
"""
map_chemicals.py - Command-line version of SpecMatch for chemical mapping

Usage:
    python map_chemicals.py --folder /path/to/data/folder
    python map_chemicals.py --folder "../TransferNow-rohdaten/marco/Lumos output exportToEnvi"
"""

import os
import sys
import argparse
import numpy as np
import yaml
import cv2
from utils import enviread, hex_to_rgb
import onnxruntime as ort

# Default model folder
MODEL_FOLDER = ".."

# Example data folder paths (for reference)
EXAMPLE_DATA_FOLDERS = {
    "Agilent": "../TransferNow-rohdaten/marco/agilent output raw+exportToEnvi",
    "Lumos": "../TransferNow-rohdaten/marco/Lumos output exportToEnvi"
}


class ChemicalMapper:
    """Non-GUI version of SpecMatch for command-line processing"""
    
    def __init__(self, measurement_folder, model_folder, force_agilent=False):
        self.measurement_folder = measurement_folder
        self.model_folder = model_folder
        self.measurement = None
        self.force_agilent = force_agilent
        self.data_type = None  # Will be detected automatically or forced
        self.image_SpecMatch = None
        self.image_labeled_SpecMatch = None
        self.contours = None
        self.particles = None
        
        # Model-related attributes (will be loaded later)
        self.substance_colors = None
        self.substance_to_idx = None
        self.idx_to_substance = None
        self.session = None
        self.model_expected_wavelengths = None
        
    def find_envi_files(self, folder_path):
        """Find .dat/.hdr or .img/.hdr file pair in the folder"""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")
        
        # Strategy 1: stitched_{basename}
        base = os.path.basename(folder_path)
        candidate = os.path.join(folder_path, f'stitched_{base}')
        for ext in ['.dat', '.img']:
            if os.path.isfile(f'{candidate}{ext}') and os.path.isfile(f'{candidate}.hdr'):
                return candidate, ext
        
        # Strategy 2: {basename}
        candidate = os.path.join(folder_path, base)
        for ext in ['.dat', '.img']:
            if os.path.isfile(f'{candidate}{ext}') and os.path.isfile(f'{candidate}.hdr'):
                return candidate, ext
        
        # Strategy 3: Find any .dat/.hdr or .img/.hdr pair
        for ext in ['.dat', '.img']:
            data_files = [f for f in os.listdir(folder_path) if f.endswith(ext)]
            for data_file in data_files:
                base_name = data_file[:-4]  # Remove extension
                hdr_file = base_name + '.hdr'
                if os.path.isfile(os.path.join(folder_path, hdr_file)):
                    file_path = os.path.join(folder_path, base_name)
                    print(f"[Measurement] Found data pair: {base_name}{ext}/.hdr")
                    return file_path, ext
        
        raise FileNotFoundError(f"No valid .dat/.hdr or .img/.hdr file pair found in {folder_path}")
    
    def load_measurement(self):
        """Load hyperspectral measurement data"""
        print(f"Loading measurement from: {self.measurement_folder}")
        
        file_path, ext = self.find_envi_files(self.measurement_folder)
        
        # Load ENVI format data (works with both .dat and .img)
        [spec_img, info1] = enviread(f'{file_path}{ext}', f'{file_path}.hdr')
        
        self.measurement = {
            "spec": np.array(spec_img), 
            "wavelengths": [int(float(w)) for w in info1.metadata["wavelength"]]
        }
        
        # Resample wavelengths if needed to match model expectations
        if self.model_expected_wavelengths is not None and len(self.measurement["wavelengths"]) != self.model_expected_wavelengths:
            self.resample_wavelengths()
        
        print(f"  Shape: {self.measurement['spec'].shape}")
        print(f"  Wavelengths: {len(self.measurement['wavelengths'])} bands")
        print(f"  Range: {self.measurement['wavelengths'][0]} - {self.measurement['wavelengths'][-1]} nm")
        
        return self.measurement
    
    def resample_wavelengths(self):
        """Resample wavelengths to match model's expected wavelength count using linear interpolation"""
        original_count = len(self.measurement['wavelengths'])
        print(f"  Resampling wavelengths from {original_count} to {self.model_expected_wavelengths}...")
        
        original_wavelengths = np.array(self.measurement['wavelengths'])
        
        # Create new wavelength array with same range but different count
        new_wavelengths = np.linspace(
            original_wavelengths[0], 
            original_wavelengths[-1], 
            self.model_expected_wavelengths
        )
        
        # Interpolate spectral data for each pixel
        spec_img = self.measurement['spec']
        new_spec = np.zeros((spec_img.shape[0], spec_img.shape[1], self.model_expected_wavelengths))
        
        for i in range(spec_img.shape[0]):
            for j in range(spec_img.shape[1]):
                # Linear interpolation for each pixel's spectrum
                new_spec[i, j] = np.interp(new_wavelengths, original_wavelengths, spec_img[i, j])
        
        # Update measurement with resampled data
        self.measurement['spec'] = new_spec
        self.measurement['wavelengths'] = new_wavelengths.tolist()

    
    def load_model(self):
        """Load model metadata and ONNX model"""
        print(f"Loading model metadata from: {self.model_folder}")
        
        metadata_path = os.path.join(self.model_folder, "metadata.yaml")
        if not os.path.isfile(metadata_path):
            print(f"  Warning: Model metadata not found at {metadata_path}")
            print(f"  Skipping model loading")
            return
        
        with open(metadata_path, 'r') as file:
            yaml_file = yaml.safe_load(file)
            self.substance_colors = yaml_file["substance_colors"]
            self.substance_to_idx = yaml_file["substance_to_idx"]
            self.idx_to_substance = yaml_file["idx_to_substance"]
        
        # Convert hex colors to RGB
        self.substance_colors = {k: hex_to_rgb(v) for k, v in self.substance_colors.items()}
        
        print(f"  Loaded {len(self.substance_colors)} substance classes:")
        for substance in self.substance_colors.keys():
            print(f"    - {substance}")
        
        # Load ONNX model
        model_path = os.path.join(self.model_folder, "model.onnx")
        if os.path.isfile(model_path):
            self.session = ort.InferenceSession(model_path)
            print(f"  Loaded ONNX model: {model_path}")
            
            # Get expected wavelength count from model input shape
            input_shape = self.session.get_inputs()[0].shape
            self.model_expected_wavelengths = input_shape[2]  # Shape is (batch, 1, wavelengths)
            print(f"  Model expects {self.model_expected_wavelengths} wavelengths")
        else:
            print(f"  Warning: ONNX model not found at {model_path}")
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def preprocessing(self, data):
        """Preprocess spectral data (placeholder - currently returns data as-is)"""
        return data
    
    def run_classification(self):
        """Run spectral classification using the ONNX model"""
        if self.session is None:
            print("\n[Classification] Skipping - no model loaded")
            return
        
        print("\n[Classification] Running AI model inference...")
        
        spec_img = self.measurement["spec"]
        wavelengths = self.measurement["wavelengths"]
        
        # Initialize output images
        self.image_SpecMatch = np.full((spec_img.shape[0], spec_img.shape[1], 3), (1.0, 1.0, 1.0))
        self.image_labeled_SpecMatch = np.full(spec_img.shape[:2], "", dtype=object)
        
        # Temporary array to build raw probabilities
        raw_probs = np.zeros((spec_img.shape[0], spec_img.shape[1], len(self.substance_colors)), dtype=float)
        
        # Process row by row
        for i in range(spec_img.shape[0]):
            # Prepare data for this row
            # spec_img[i] has shape (width, num_wavelengths)
            # We need shape (width, 1, num_wavelengths)
            data = np.expand_dims(self.preprocessing(spec_img[i]), axis=1).astype(np.float32)
            
            # Run inference
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            outputs = self.session.run([output_name], {input_name: data})[0]
            
            # Get predictions
            predicted = np.argmax(outputs, axis=1)
            predicted = [self.idx_to_substance[int(p)] for p in predicted]
            
            # Store results
            for j, p in enumerate(predicted):
                raw_probs[i, j] = self.softmax(outputs[j])
                self.image_SpecMatch[i, j] = self.substance_colors[p]
                if p != "Noise" and p != "noise":
                    self.image_labeled_SpecMatch[i, j] = p
            
            # Progress indicator
            if (i + 1) % 10 == 0 or i == spec_img.shape[0] - 1:
                print(f"  Processed {i + 1}/{spec_img.shape[0]} rows")
        
        self.probabilities_raw = raw_probs
        print(f"  Classification complete")
        
        # Count predictions
        unique, counts = np.unique(self.image_labeled_SpecMatch[self.image_labeled_SpecMatch != ""], return_counts=True)
        print(f"\n  Predicted substances:")
        for substance, count in zip(unique, counts):
            print(f"    - {substance}: {count} pixels")
    
    def denoise(self, kernel_size=2, min_area=10, max_area=float("inf")):
        """Denoise the classified image (placeholder)"""
        print(f"\n[Denoising] Skipping (requires classification first)")
        # This will be implemented after classification is working
    
    def group_particles(self):
        """Group particles based on contours (placeholder)"""
        print(f"\n[Grouping] Skipping (requires classification first)")
        # This will be implemented after classification is working
    
    def save_classification_image(self, output_folder="../output"):
        """Save the classification image and raw data to files"""
        if self.image_SpecMatch is None:
            print("\n[Save Results] No classification data to save")
            return
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate filename based on measurement folder name
        measurement_name = os.path.basename(self.measurement_folder)
        if not measurement_name:  # Handle case where folder path ends with /
            measurement_name = os.path.basename(os.path.dirname(self.measurement_folder))
        
        # Save PNG image
        png_path = os.path.join(output_folder, f"{measurement_name}_classified.png")
        image_bgr = (self.image_SpecMatch[..., ::-1] * 255).astype(np.uint8)
        cv2.imwrite(png_path, image_bgr)
        print("\n[Save Results]")
        print(f"  PNG image saved to: {png_path}")
        
        # Save raw classification data (RGB values, 0-1 range)
        rgb_data_path = os.path.join(output_folder, f"{measurement_name}_classified_rgb.npy")
        np.save(rgb_data_path, self.image_SpecMatch)
        print(f"  RGB data saved to: {rgb_data_path}")
        
        # Save labeled classification data (substance names)
        labels_path = os.path.join(output_folder, f"{measurement_name}_classified_labels.npy")
        np.save(labels_path, self.image_labeled_SpecMatch)
        print(f"  Label data saved to: {labels_path}")
        
        # Save probability data if available
        if hasattr(self, 'probabilities_raw') and self.probabilities_raw is not None:
            prob_path = os.path.join(output_folder, f"{measurement_name}_probabilities.npy")
            np.save(prob_path, self.probabilities_raw)
            print(f"  Probability data saved to: {prob_path}")
        
        # Save metadata as text file for easy reference
        metadata_path = os.path.join(output_folder, f"{measurement_name}_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write("Classification Results Metadata\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Measurement folder: {self.measurement_folder}\n")
            f.write(f"Data type: {self.data_type if self.data_type else 'Not detected'}\n")
            f.write(f"Image shape: {self.image_SpecMatch.shape}\n")
            f.write(f"Model folder: {self.model_folder}\n\n")
            
            # Count substances
            unique_labels = np.unique(self.image_labeled_SpecMatch[self.image_labeled_SpecMatch != ""])
            f.write(f"Detected substances:\n")
            for label in unique_labels:
                count = np.sum(self.image_labeled_SpecMatch == label)
                f.write(f"  - {label}: {count} pixels\n")
            
            f.write(f"\nFile descriptions:\n")
            f.write(f"  - *_classified.png: Visual classification image (PNG format)\n")
            f.write(f"  - *_classified_rgb.npy: Raw RGB data (numpy array, shape: {self.image_SpecMatch.shape})\n")
            f.write(f"  - *_classified_labels.npy: Substance labels (numpy array, shape: {self.image_labeled_SpecMatch.shape})\n")
            if hasattr(self, 'probabilities_raw') and self.probabilities_raw is not None:
                f.write(f"  - *_probabilities.npy: Classification probabilities (numpy array, shape: {self.probabilities_raw.shape})\n")
            f.write(f"\nTo load the data in Python:\n")
            f.write(f"  import numpy as np\n")
            f.write(f"  rgb_data = np.load('{os.path.basename(rgb_data_path)}')\n")
            f.write(f"  labels = np.load('{os.path.basename(labels_path)}', allow_pickle=True)\n")
            if hasattr(self, 'probabilities_raw') and self.probabilities_raw is not None:
                f.write(f"  probabilities = np.load('{os.path.basename(prob_path)}')\n")
        
        print(f"  Metadata saved to: {metadata_path}")
        print(f"\n  Total files saved: {4 if hasattr(self, 'probabilities_raw') and self.probabilities_raw is not None else 3} + metadata")
        
        return png_path
    
    def export_results(self, output_folder=None):
        """Export results to CSV (placeholder)"""
        if output_folder is None:
            output_folder = self.measurement_folder
        
        print(f"\n[Export] Skipping (requires classification first)")
        print(f"  Results would be saved to: {output_folder}")
        # This will be implemented after classification is working
    
    def run_pipeline(self):
        """Execute the full processing pipeline"""
        print("="*60)
        print("SpecMatch Chemical Mapping - Command Line Version")
        print("="*60)
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Load measurement data
        self.load_measurement()
        
        
        # Step 3: Run classification
        self.run_classification()
        
        # Step 4: Save classification results (image + raw data)
        self.save_classification_image()
        
        # Step 5: Denoise (placeholder)
        # self.denoise()
        
        # Step 6: Group particles (placeholder)
        # self.group_particles()
        
        # Step 7: Export results (placeholder)
        # self.export_results()
        
        print("\n" + "="*60)
        print("Processing complete!")
        print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SpecMatch Chemical Mapping - Command Line Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect or use Lumos (default):
  python map_chemicals.py --folder "../TransferNow-rohdaten/marco/Lumos output exportToEnvi"
  
  # Force Agilent processing:
  python map_chemicals.py --folder "../TransferNow-rohdaten/marco/agilent output raw+exportToEnvi" --Agilent
  
  # Auto-detect from any folder:
  python map_chemicals.py --folder /path/to/your/data
        """
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        required=True,
        help='Path to the measurement data folder containing ENVI format files (.dat/.hdr or .img/.hdr)'
    )
    
    parser.add_argument(
        '--Agilent',
        action='store_true',
        help='Force Agilent processing mode (default: auto-detect or use Lumos)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_FOLDER,
        help=f'Path to the model folder (default: {MODEL_FOLDER})'
    )
    
    args = parser.parse_args()
    
    # Validate that the folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: Folder not found: {args.folder}")
        return 1
    
    measurement_folder = args.folder
    model_folder = args.model
    
    print(f"Measurement folder: {measurement_folder}")
    print(f"Model folder: {model_folder}")
    if args.Agilent:
        print("Processing mode: Agilent (forced)")
    else:
        print("Processing mode: Auto-detect or Lumos (default)")
    print()
    
    mapper = ChemicalMapper(
        measurement_folder=measurement_folder,
        model_folder=model_folder,
        force_agilent=args.Agilent
    )
    
    try:
        mapper.run_pipeline()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
