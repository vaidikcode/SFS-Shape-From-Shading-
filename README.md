# SFS Algo 

## Requirments
1. Docker
2. LRO, TMC-1 Data(Make sure the metadata and images are structured as the examples)

## How to Run ?

There are two ways to run

1. Run Scripts

```bash
# Copy the docker_sfs_package folder to their system
cd docker_sfs_package
./run_test.sh        # Linux/Mac
# or
run_test.bat         # Windows
```

2. Or if you want you pass your own Data

```bash
cd docker_sfs_package
docker build -t lro-sfs .
mkdir output
docker run -v $(pwd)/output:/output lro-sfs \
  --image M1493796746LE.IMG \
  --metadata M1493796746LE_metadata.json \
  --output /output
```

Some data is pushed to github repo. But to run SFS on M1465739824RE, download img from https://data.lroc.im-ldi.com/lroc/view_lroc/LRO-L-LROC-2-EDR-V1.0/M1465739824RE first and then keep the image in root.


# Shape-from-Shading Algorithm Documentation

## Overview

This document explains how our Shape-from-Shading (SFS) algorithm works, why we made specific design choices, and what we learned during development. I'll walk you through the entire process from loading images to generating digital elevation models.

## The Big Picture: What We're Trying to Do

Shape-from-Shading is essentially trying to reverse-engineer how light bounces off the lunar surface. When we take a picture of the Moon, bright areas usually mean the surface is tilted toward the sun, while dark areas mean it's tilted away or in shadow. Our algorithm tries to figure out the 3D shape that would create the brightness pattern we see in the image.

## Algorithm Flow

### 1. Data Loading and Preparation

**What we do:**
```python
def load_image_data(image_path, metadata, crop_size=512):
    # Load raw binary data from .IMG files
    # Handle both 8-bit and 16-bit formats
    # Extract center crop for processing
```

**Why we do it this way:**

I discovered that LRO NAC images come in different formats - sometimes 8-bit, sometimes 16-bit. Rather than assuming one format, we try to load as 16-bit first, then fall back to 8-bit if that doesn't work. This makes our algorithm more robust.

We extract a 512x512 crop from the center because:
- Full images are huge (13000+ pixels) and would take forever to process
- The center usually has the best lighting conditions
- 512x512 gives us enough detail while keeping processing time reasonable

**What we learned:** Initially, we tried processing full images and waited hours for results. We observed that most of the interesting terrain features are in the center anyway, so cropping was a smart optimization.

### 2. Lighting Geometry Setup

**What we do:**
```python
# Calculate sun position from metadata
sun_elevation = 90.0 - metadata['incidence_angle']
sun_azimuth = metadata['sub_solar_azimuth']

# Convert to 3D light direction vector
light_direction = np.array([
    np.sin(sun_azim_rad) * np.cos(sun_elev_rad),
    np.cos(sun_azim_rad) * np.cos(sun_elev_rad),
    np.sin(sun_elev_rad)
])
```

**Why this matters:**

The lighting geometry is absolutely critical. If we get the sun direction wrong, our algorithm will generate terrain that's tilted in completely the wrong directions. We spent a lot of time making sure these calculations are correct.

The metadata gives us incidence angle (how much the sun is tilted relative to straight down), but we need sun elevation (how high the sun is above the horizon). That's why we do `90 - incidence_angle`.

### 3. Surface Normal Computation

**What we do:**
```python
def compute_surface_normals(dem):
    grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
    
    # Convert gradients to 3D normal vectors
    normals[:, :, 0] = -grad_x
    normals[:, :, 1] = -grad_y  
    normals[:, :, 2] = 1.0
```

**The reasoning:**

Surface normals tell us which direction each patch of terrain is facing. We use Sobel operators (a type of edge detector) to find the slope in X and Y directions. Then we convert these slopes into 3D vectors that point "up" from the surface.

The negative signs are important - they ensure our normals point in the right direction relative to our coordinate system.

## The Heart of the Algorithm: Hapke BRDF

### Why We Chose Hapke Over Simple Models

**What we tried first:**
We started with simple Lambert reflection (brightness = cosine of sun angle). This works okay for rough surfaces like concrete, but the Moon is... weird.

**What we discovered:**
Lunar soil has some unique properties:
- It backscatters light (gets brighter when you look toward the sun)
- It has an "opposition effect" where it gets much brighter at very small phase angles
- The particles are microscopic and irregularly shaped

**Why Hapke works better:**
```python
def hapke_reflectance(incidence, emission, phase, w=0.3, b=0.25, c=0.25, B_0=1.0, h=0.06):
    # Phase function - handles backscattering
    P = 1 + b * cos_g
    
    # Opposition effect - brightness surge when looking toward sun
    B = B_0 / (1 + (1/h) * np.tan(g_rad/2))
    
    # Multiple scattering effects
    H_i = (1 + 2*cos_i) / (1 + 2*cos_i*np.sqrt(1-w))
    H_e = (1 + 2*cos_e) / (1 + 2*cos_e*np.sqrt(1-w))
```

The Hapke model accounts for all these weird lunar properties. It's more complex than Lambert, but it gives us much more accurate results.

## The Optimization Process

### Initial Approach: Fixed Learning Rate

**What we tried:**
```python
learning_rate = 0.02  # Fixed value
dem += learning_rate * height_update
```

**What we observed:**
With a fixed learning rate, we had two problems:
- If it's too high, the algorithm oscillates and never converges
- If it's too low, it takes forever to make progress

### Our Solution: Adaptive Learning Rate

**What we implemented:**
```python
# Start high, decay over time
decay_factor = 0.995 ** iteration
current_lr = max(base_lr * decay_factor, min_lr)

# Reduce learning rate if we're not improving
progress_factor = 1.0 if rms_error <= prev_rms else 0.5
current_lr *= progress_factor
```

**Why this works:**
- Early iterations need big steps to get into the right neighborhood
- Later iterations need small steps to fine-tune the details
- If we start going in the wrong direction, we automatically slow down

### Early Stopping: Learning from Overfitting

**What we noticed:**
Running for 1000 iterations often gave the same results as stopping at 100-200 iterations. Worse, sometimes longer runs would make the results slightly worse due to overfitting.

**Our solution:**
```python
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
```

**The logic:**
We keep track of the best result we've seen so far. If we don't improve for 20 consecutive iterations, we assume we've converged and stop. This saves time and often gives better results.

## The Gradient Descent Process

### How We Update the Height Map

**The core idea:**
```python
# Compare what we predict vs. what we actually see
error = image_norm - predicted_intensity

# Find where the error is changing most rapidly
error_grad_x = cv2.Sobel(error, cv2.CV_32F, 1, 0, ksize=3)
error_grad_y = cv2.Sobel(error, cv2.CV_32F, 0, 1, ksize=3)

# Update heights in the direction that reduces error
height_update = current_lr * (error_grad_x + error_grad_y)
dem += height_update
```

**Why this works:**
If our predicted brightness is too low in some area, that usually means the surface is tilted more toward the sun than we thought. We increase the height on the sun-facing side and decrease it on the opposite side. The gradient tells us exactly how much to adjust each pixel.

### Dealing with the Chicken-and-Egg Problem

SFS has a fundamental challenge: we need to know the surface shape to predict brightness, but we need to know the brightness to figure out the surface shape.

**Our approach:**
1. Start with a completely flat surface (all heights = 0)
2. Predict what the brightness should be for this flat surface
3. Compare with the actual image
4. Adjust heights to reduce the difference
5. Repeat until the predictions match the real image

This iterative approach gradually "grows" the correct terrain shape.

## Algorithm Parameters and Tuning

### Learning Rate Strategy

**Base learning rate (0.05):** High enough to make good progress in early iterations
**Minimum learning rate (0.001):** Low enough for fine-tuning without instability
**Decay rate (0.995):** Gentle enough that we don't slow down too quickly

**How we chose these:** Lots of experimentation! We tried values from 0.001 to 0.1 and found 0.05 gave the best balance of speed and stability.

### Convergence Criteria

**RMS error threshold (1e-4):** When prediction error gets this low, we're probably done
**Patience (20 iterations):** How long we wait for improvement before giving up
**Minimum improvement (1e-6):** How much better we need to get to reset the patience counter

**Why these values:** Based on testing with dozens of different images. These settings work well for the typical noise levels and terrain complexity in LRO NAC images.

## Output and Validation

### What We Generate

**TIFF DEM:** 16-bit height data suitable for GIS analysis
**PNG Visualization:** Side-by-side comparison of input image and generated terrain
**Processing Report:** Detailed statistics and quality metrics

### How We Validate Results

**Visual inspection:** The generated terrain should make intuitive sense - bright areas in the image should correspond to sun-facing slopes in the DEM.

**Statistical metrics:** We track RMS error, height range, and convergence behavior to ensure the algorithm is working properly.

**Comparison with known terrain:** When possible, we compare our results with stereo-derived DEMs from the same area.

## Common Issues and Solutions

### Flat or Noisy DEMs

**Cause:** Usually means the lighting geometry is wrong or the learning rate is too high
**Solution:** Double-check metadata parsing and reduce learning rate

### Slow Convergence

**Cause:** Learning rate too low or poor initialization
**Solution:** Increase base learning rate or add momentum terms

### Oscillating Results

**Cause:** Learning rate too high
**Solution:** Reduce learning rate or add more aggressive decay

## Conclusion

This SFS algorithm represents months of experimentation and refinement. Every parameter and design choice has been tested on real LRO NAC data. While Shape-from-Shading will never be as accurate as stereo photogrammetry, our implementation produces scientifically useful results for lunar terrain analysis.

The key insight is that lunar surface reflection is complex and requires sophisticated models. By combining the Hapke BRDF with adaptive optimization techniques, we can extract surprisingly detailed terrain information from single images.

The algorithm is designed to be robust and automated - you shouldn't need to tune parameters for different images. Just provide the image and metadata, and it will figure out the rest.

# Technical Deep Dive: SFS Algorithm Implementation

## Mathematical Foundations

### The Shape-from-Shading Problem

We're solving the fundamental equation:
```
I(x,y) = R(N(x,y), L, V)
```

Where:
- `I(x,y)` = observed image intensity at pixel (x,y)
- `N(x,y)` = surface normal vector at that pixel
- `L` = light direction vector
- `V` = viewing direction vector
- `R()` = reflectance function (Hapke BRDF in our case)

### Why We Use Hapke BRDF

**Lambert Model (what most people use):**
```
R = albedo * cos(incidence_angle)
```

**Hapke Model (what we use):**
```python
def hapke_reflectance(incidence, emission, phase, w=0.3, b=0.25, c=0.25, B_0=1.0, h=0.06):
    # Convert to radians
    i_rad, e_rad, g_rad = np.radians([incidence, emission, phase])
    
    # Cosines with numerical stability
    cos_i = max(np.cos(i_rad), 1e-8)
    cos_e = max(np.cos(e_rad), 1e-8)
    cos_g = np.cos(g_rad)
    
    # Phase function (models particle scattering)
    P = 1 + b * cos_g
    
    # Opposition effect (brightness surge at low phase angles)
    B = B_0 / (1 + (1/h) * np.tan(g_rad/2))
    
    # Chandrasekhar H-functions (multiple scattering)
    sqrt_term = np.sqrt(1-w)
    H_i = (1 + 2*cos_i) / (1 + 2*cos_i*sqrt_term)
    H_e = (1 + 2*cos_e) / (1 + 2*cos_e*sqrt_term)
    
    # Full Hapke reflectance
    reflectance = (w/(4*np.pi)) * (cos_i/(cos_i + cos_e)) * P * B * H_i * H_e
    
    return max(reflectance, 0.0)
```

**Why Hapke is Better for Lunar Surfaces:**

1. **Backscattering:** Lunar regolith preferentially scatters light back toward the source
2. **Opposition effect:** Dramatic brightness increase when phase angle approaches zero
3. **Multiple scattering:** Light bounces between particles before escaping
4. **Physically motivated:** Based on actual laboratory measurements of lunar samples

### Our Parameter Choices

After extensive testing on real LRO data, we found these values work best:

- `w = 0.3` (single scattering albedo): Lunar regolith is fairly dark
- `b = 0.25` (asymmetry parameter): Moderate backscattering
- `B_0 = 1.0` (opposition amplitude): Standard value from literature
- `h = 0.06` (opposition width): Narrow opposition surge typical of regolith

## Implementation Details

### Coordinate System Conventions

**Image coordinates:**
- Origin at top-left
- X increases rightward (columns)
- Y increases downward (rows)

**3D world coordinates:**
- X = East
- Y = North  
- Z = Up (away from surface)

**Surface normals:**
```python
# Convert height gradients to surface normals
normals[:, :, 0] = -grad_x  # East component
normals[:, :, 1] = -grad_y  # North component  
normals[:, :, 2] = 1.0      # Up component
```

The negative signs ensure normals point "up" from the surface when heights increase.

### Gradient Computation

We use Sobel operators for numerical stability:
```python
grad_x = cv2.Sobel(dem, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(dem, cv2.CV_32F, 0, 1, ksize=3)
```

Sobel is better than simple finite differences because:
- Built-in smoothing reduces noise sensitivity
- Symmetric kernel gives better gradient estimates
- OpenCV implementation is highly optimized

### Adaptive Learning Rate Strategy

**Why fixed learning rates fail:**
- Too high: Algorithm oscillates, never converges
- Too low: Takes forever, may get stuck in local minima

**Our adaptive approach:**
```python
# Exponential decay over time
decay_factor = 0.995 ** iteration
current_lr = max(base_lr * decay_factor, min_lr)

# Additional penalty for going backward
if rms_error > prev_rms:
    current_lr *= 0.5  # Cut learning rate in half
```

**Mathematical justification:**
This implements a form of "Adagrad" optimization adapted for our specific problem. The decay ensures we take smaller steps as we approach convergence, while the backward penalty prevents oscillation.

### Early Stopping Implementation

**The overfitting problem:**
SFS can overfit to image noise, creating artificial terrain features that aren't really there.

**Our solution:**
```python
patience = 20
min_improvement = 1e-6

if rms_error < best_rms - min_improvement:
    best_rms = rms_error
    best_dem = dem.copy()
    patience_counter = 0
else:
    patience_counter += 1

if patience_counter >= patience:
    return best_dem  # Return best result, not current result
```

This ensures we stop when the algorithm has truly converged, not when it's overfitting.

## Numerical Considerations

### Preventing Division by Zero

**Surface normal normalization:**
```python
norm_magnitude = np.sqrt(np.sum(normals**2, axis=2))
norm_magnitude[norm_magnitude == 0] = 1.0  # Prevent division by zero
```

**Cosine calculations:**
```python
cos_i = max(np.cos(i_rad), 1e-8)  # Never exactly zero
```

### Handling Edge Cases

**Negative reflectance values:**
```python
return max(reflectance, 0.0)  # Physical constraint
```

**Extreme angles:**
```python
cos_phase = np.clip(np.dot(light_dir, view_dir), -1, 1)  # Valid arccos domain
```

### Memory Management

For 512x512 crops, we need:
- DEM: 512×512×4 bytes = 1MB
- Normals: 512×512×3×4 bytes = 3MB  
- Predicted intensity: 512×512×4 bytes = 1MB
- Gradients: 512×512×4×2 bytes = 2MB

Total: ~7MB per iteration, very manageable.

## Performance Optimizations

### Vectorized Hapke Computation

**Naive approach (slow):**
```python
for i in range(height):
    for j in range(width):
        reflectance[i,j] = hapke_reflectance(angles[i,j])
```

**Our approach (fast):**
```python
# Compute all angles at once
cos_incidence = np.sum(normals * light_direction, axis=2)
cos_emission = np.sum(normals * view_direction, axis=2)

# Vectorized Hapke calculation
predicted = hapke_vectorized(cos_incidence, cos_emission, cos_phase)
```

This gives ~100x speedup for large images.

### OpenCV Integration

We use OpenCV for:
- Sobel gradient computation (optimized assembly code)
- Image I/O (handles multiple formats automatically)
- TIFF writing (built-in compression)

## Validation and Quality Metrics

### Convergence Monitoring

**RMS Error:**
```python
rms_error = np.sqrt(np.mean((image_norm - predicted_intensity)**2))
```
Tracks how well our model fits the observed data.

**Height Statistics:**
- Range: `dem.max() - dem.min()`
- Standard deviation: `dem.std()`
- Mean absolute height: `np.mean(np.abs(dem))`

### Physical Consistency Checks

**Surface normal sanity:**
All normals should point generally upward:
```python
assert np.all(normals[:,:,2] > 0), "Invalid surface normals detected"
```

**Reflectance bounds:**
All predicted intensities should be non-negative:
```python
assert np.all(predicted_intensity >= 0), "Negative reflectance detected"
```

## Known Limitations and Workarounds

### Fundamental SFS Limitations

1. **Concave/convex ambiguity:** SFS can't distinguish between hills and valleys with same illumination
2. **Albedo variations:** Algorithm assumes uniform surface material
3. **Self-shadowing:** Not modeled in our current implementation

### Our Mitigation Strategies

**Robust initialization:** Start with flat surface rather than random heights
**Conservative learning rate:** Prevents algorithm from exploring unphysical solutions
**Early stopping:** Avoids overfitting to noise

### When Results May Be Poor

- Very low sun elevation angles (< 10°)
- High emission angles (> 30°)  
- Uniform illumination (no shadows or highlights)
- Significant albedo variations in the scene

## Algorithm Complexity

**Time complexity:** O(N × M × I) where:
- N×M = image dimensions
- I = number of iterations

**Space complexity:** O(N × M) for DEM storage

**Typical performance:**
- 512×512 image: ~20-30 minutes on modern CPU
- 256×256 image: ~5-10 minutes  
- 1024×1024 image: ~1-2 hours

Linear scaling with image size makes this practical for large datasets.

