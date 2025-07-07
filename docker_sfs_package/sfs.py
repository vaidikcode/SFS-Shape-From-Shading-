#!/usr/bin/env python3

import numpy as np
import json
import cv2
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LROMetadata:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.data = self._load_metadata()
        self._extract_metadata()
        
    def _load_metadata(self):
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata from {self.metadata_path}: {e}")
            raise
    
    def _extract_metadata(self):
        if 'geometry' in self.data:
            geom = self.data['geometry']
            self.data.update({
                'incidence_angle': geom.get('incidence_angle'),
                'sub_solar_azimuth': geom.get('sub_solar_azimuth'),
                'emission_angle': geom.get('emission_angle'),
                'phase_angle': geom.get('phase_angle')
            })
        
        if 'image_properties' in self.data:
            img_props = self.data['image_properties']
            self.data.update({
                'resolution': img_props.get('resolution'),
                'image_lines': img_props.get('image_lines'),
                'line_samples': img_props.get('line_samples')
            })
    
    @property
    def sun_elevation(self):
        return 90.0 - self.data['incidence_angle']
    
    @property
    def sun_azimuth(self):
        return self.data['sub_solar_azimuth']
    
    @property
    def emission_angle(self):
        return self.data['emission_angle']
    
    @property
    def phase_angle(self):
        return self.data['phase_angle']
    
    @property
    def resolution(self):
        return self.data['resolution']
    
    @property
    def image_dimensions(self):
        return (self.data['image_lines'], self.data['line_samples'])

class LunarReflectanceModel:
    @staticmethod
    def hapke_reflectance(incidence, emission, phase, w=0.3, b=0.25, c=0.25, B_0=1.0, h=0.06):
        i_rad = np.radians(incidence)
        e_rad = np.radians(emission)
        g_rad = np.radians(phase)
        
        cos_i = max(np.cos(i_rad), 1e-8)
        cos_e = max(np.cos(e_rad), 1e-8)
        
        cos_g = np.cos(g_rad)
        P = 1 + b * cos_g
        
        B = B_0 / (1 + (1/h) * np.tan(g_rad/2))
        
        H_i = (1 + 2*cos_i) / (1 + 2*cos_i*np.sqrt(1-w))
        H_e = (1 + 2*cos_e) / (1 + 2*cos_e*np.sqrt(1-w))
        
        reflectance = (w/(4*np.pi)) * (cos_i/(cos_i + cos_e)) * P * B * H_i * H_e
        
        return max(reflectance, 0.0)

def load_image_data(image_path, metadata, crop_size=512):
    rows = metadata['image_lines']
    cols = metadata['line_samples']
    
    logger.info(f"Loading image: {image_path} ({rows}x{cols})")
    
    with open(image_path, 'rb') as f:
        data = f.read()
    
    expected_16bit = rows * cols * 2
    expected_8bit = rows * cols
    
    if len(data) >= expected_16bit:
        try:
            raw_data = np.frombuffer(data[-expected_16bit:], dtype=np.uint16)
            image = raw_data.reshape(rows, cols)
        except:
            raw_data = np.frombuffer(data[-expected_8bit:], dtype=np.uint8)
            image = raw_data.reshape(rows, cols)
    else:
        raw_data = np.frombuffer(data[-expected_8bit:], dtype=np.uint8)
        image = raw_data.reshape(rows, cols)
    
    center_row, center_col = rows // 2, cols // 2
    half_crop = crop_size // 2
    
    cropped = image[center_row-half_crop:center_row+half_crop, 
                   center_col-half_crop:center_col+half_crop]
    
    return cropped.astype(np.float32), (center_row-half_crop, center_col-half_crop)

def compute_surface_normals(dem):
    grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
    
    height, width = dem.shape
    normals = np.zeros((height, width, 3), dtype=np.float32)
    normals[:, :, 0] = -grad_x
    normals[:, :, 1] = -grad_y  
    normals[:, :, 2] = 1.0
    
    norm_magnitude = np.sqrt(np.sum(normals**2, axis=2))
    norm_magnitude[norm_magnitude == 0] = 1.0
    
    normals[:, :, 0] /= norm_magnitude
    normals[:, :, 1] /= norm_magnitude
    normals[:, :, 2] /= norm_magnitude
    
    return normals

def predict_image_intensity(normals, light_dir, view_dir, reflectance_model):
    height, width = normals.shape[:2]
    predicted = np.zeros((height, width), dtype=np.float32)
    
    for i in range(height):
        for j in range(width):
            normal = normals[i, j, :]
            
            cos_incidence = np.dot(normal, light_dir)
            cos_emission = np.dot(normal, view_dir)
            cos_phase = np.dot(light_dir, view_dir)
            
            incidence_angle = np.degrees(np.arccos(np.clip(cos_incidence, -1, 1)))
            emission_angle = np.degrees(np.arccos(np.clip(cos_emission, -1, 1)))
            phase_angle = np.degrees(np.arccos(np.clip(cos_phase, -1, 1)))
            
            predicted[i, j] = reflectance_model.hapke_reflectance(
                incidence_angle, emission_angle, phase_angle
            )
    
    return predicted

def shape_from_shading(image, metadata, max_iterations=1000):
    height, width = image.shape
    logger.info(f"Starting SFS on {height}x{width} image")
    
    image_norm = image / np.max(image)
    
    sun_elevation = 90.0 - metadata['incidence_angle']
    sun_azimuth = metadata['sub_solar_azimuth']
    emission_angle = metadata['emission_angle']
    
    sun_elev_rad = np.radians(sun_elevation)
    sun_azim_rad = np.radians(sun_azimuth)
    emission_rad = np.radians(emission_angle)
    
    light_direction = np.array([
        np.sin(sun_azim_rad) * np.cos(sun_elev_rad),
        np.cos(sun_azim_rad) * np.cos(sun_elev_rad),
        np.sin(sun_elev_rad)
    ])
    
    view_direction = np.array([
        np.sin(emission_rad),
        0.0,
        np.cos(emission_rad)
    ])
    
    dem = np.zeros((height, width), dtype=np.float32)
    reflectance_model = LunarReflectanceModel()
    
    base_lr = 0.05
    min_lr = 0.001
    patience = 20
    min_improvement = 1e-6
    
    best_rms = float('inf')
    best_dem = None
    patience_counter = 0
    prev_rms = float('inf')
    
    start_time = datetime.now()
    
    for iteration in range(max_iterations):
        normals = compute_surface_normals(dem)
        predicted_intensity = predict_image_intensity(normals, light_direction, view_direction, reflectance_model)
        
        error = image_norm - predicted_intensity
        
        error_grad_x = cv2.Sobel(error, cv2.CV_32F, 1, 0, ksize=3)
        error_grad_y = cv2.Sobel(error, cv2.CV_32F, 0, 1, ksize=3)
        
        decay_factor = 0.995 ** iteration
        progress_factor = 1.0 if iteration == 0 or np.sqrt(np.mean(error**2)) <= prev_rms else 0.5
        
        current_lr = max(base_lr * decay_factor * progress_factor, min_lr)
        
        height_update = current_lr * (error_grad_x + error_grad_y)
        dem += height_update
        
        rms_error = np.sqrt(np.mean(error**2))
        
        if iteration % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = elapsed * max_iterations / (iteration + 1) - elapsed
            logger.info(f"Iter {iteration:4d}: RMS={rms_error:.6f}, LR={current_lr:.6f}, ETA={eta/60:.1f}min")
        
        if rms_error < best_rms - min_improvement:
            best_rms = rms_error
            best_dem = dem.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at iteration {iteration}")
            dem = best_dem
            break
        
        if rms_error < 1e-4:
            logger.info(f"Converged at iteration {iteration}")
            break
        
        prev_rms = rms_error
    
    processing_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"SFS completed in {processing_time/60:.1f} minutes")
    
    return dem, processing_time

def save_results(dem, image, metadata, output_dir, processing_time):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dem_path = os.path.join(output_dir, f"SFS_DEM_{timestamp}.tif")
    png_path = os.path.join(output_dir, f"SFS_Results_{timestamp}.png")
    report_path = os.path.join(output_dir, f"sfs_report_{timestamp}.txt")
    
    try:
        dem_normalized = ((dem - dem.min()) / (dem.max() - dem.min()) * 65535).astype(np.uint16)
        cv2.imwrite(dem_path, dem_normalized)
        logger.info(f"DEM saved: {dem_path}")
    except Exception as e:
        logger.warning(f"TIFF save failed: {e}")
        dem_path = None
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f'LRO NAC Image\n{image.shape[0]}x{image.shape[1]} pixels')
        ax1.axis('off')
        
        im = ax2.imshow(dem, cmap='terrain')
        ax2.set_title(f'SFS DEM\nRange: [{dem.min():.4f}, {dem.max():.4f}]')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, label='Relative Height')
        
        fig.suptitle(f'LRO NAC SFS Results - {timestamp}\nProcessing: {processing_time/60:.1f} min', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results image saved: {png_path}")
        
    except ImportError:
        logger.warning("Matplotlib not available")
        png_path = None
    except Exception as e:
        logger.warning(f"Image save failed: {e}")
        png_path = None
    
    with open(report_path, 'w') as f:
        f.write("LRO NAC Shape-from-Shading Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Product: {metadata.get('product', 'N/A')}\n")
        f.write(f"Processing Time: {processing_time/60:.1f} minutes\n")
        f.write(f"Image Size: {dem.shape[0]} x {dem.shape[1]} pixels\n")
        f.write(f"Resolution: {metadata['resolution']:.3f} m/pixel\n\n")
        
        f.write("Illumination:\n")
        f.write(f"  Sun Elevation: {90.0 - metadata['incidence_angle']:.2f}°\n")
        f.write(f"  Sun Azimuth: {metadata['sub_solar_azimuth']:.2f}°\n")
        f.write(f"  Incidence: {metadata['incidence_angle']:.2f}°\n")
        f.write(f"  Emission: {metadata['emission_angle']:.2f}°\n\n")
        
        f.write("DEM Statistics:\n")
        f.write(f"  Min Height: {dem.min():.6f}\n")
        f.write(f"  Max Height: {dem.max():.6f}\n")
        f.write(f"  Mean: {dem.mean():.6f}\n")
        f.write(f"  Std Dev: {dem.std():.6f}\n")
        f.write(f"  Range: {dem.max() - dem.min():.6f}\n")
    
    logger.info(f"Report saved: {report_path}")
    
    return {
        'dem_tiff': dem_path,
        'png_results': png_path,
        'report': report_path,
        'processing_time': processing_time
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="LRO NAC Shape-from-Shading")
    parser.add_argument("--image", required=True, help="LRO NAC image file (.IMG)")
    parser.add_argument("--metadata", required=True, help="Metadata JSON file")
    parser.add_argument("--output", default="/output", help="Output directory")
    parser.add_argument("--crop-size", type=int, default=512, help="Crop size")
    parser.add_argument("--max-iterations", type=int, default=1000, help="Max iterations")
    
    args = parser.parse_args()
    
    logger.info("LRO NAC Shape-from-Shading Algorithm")
    logger.info("=" * 60)
    
    if not os.path.exists(args.image):
        logger.error(f"Image file not found: {args.image}")
        return 1
    
    if not os.path.exists(args.metadata):
        logger.error(f"Metadata file not found: {args.metadata}")
        return 1
    
    try:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Processing: {metadata['product']}")
        
        image, crop_offset = load_image_data(args.image, metadata, crop_size=args.crop_size)
        
        dem, processing_time = shape_from_shading(image, metadata, max_iterations=args.max_iterations)
        
        results = save_results(dem, image, metadata, args.output, processing_time)
        
        logger.info("SFS COMPLETED!")
        logger.info(f"Processing time: {processing_time/60:.1f} minutes")
        logger.info(f"DEM range: [{dem.min():.6f}, {dem.max():.6f}]")
        
        return 0
        
    except Exception as e:
        logger.error(f"SFS failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
