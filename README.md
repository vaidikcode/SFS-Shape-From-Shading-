# SFS-Shape-From-Shading-

The SFS Algo is divided below in Different Parts. At the Last We Stick it and provide a sequential flow of Mono Image and MetaData through Algo.

## Shadow Masking Function

```python
def mask_shadows(image, threshold=0.15):
    image_norm = image / np.nanmax(image)
    mask = image_norm < threshold
    image_masked = image.copy()
    image_masked[mask] = np.nan
    return image_masked, mask
```
This function takes image and threshold as input.   

`image_norm = image / np.nanmax(image`

Dividing each pixel by the max normalises it to a range from [0,1] Then We copy the image. Set all the numders of image array with mask vaalues to be "Not a Number" The new image ignoring the shadows is returned.

*This part is solely to handle the points with shadows*

## Compute Light Vector

```python
def compute_light_vector(azimuth_deg, incidence_angle_deg):
    azimuth = np.radians(azimuth_deg)
    incidence = np.radians(incidence_angle_deg)
    Lx = np.cos(azimuth) * np.sin(incidence)
    Ly = np.sin(azimuth) * np.sin(incidence)
    Lz = np.cos(incidence)
    return np.array([Lx, Ly, Lz])
```

This takes two parameters 

`azimuth_deg:` direction of the Sun across the surface (in degrees) — like compass direction

`incidence_angle_deg:` how high or low the Sun is in the sky (0° = overhead, 90° = horizon)                                                                                                       Returns  Array of direction `([Lx, Ly, Lz])` 

*This function converts the Sun's direction (given in degrees) into a 3D vector called the light direction vector, often written as L = (Lx, Ly, Lz).*

## lambertian reflectance

```python
def lambertian_reflectance(image, light_vector, max_iterations=100, lambda_smooth=0.1):
```

What this function does is to build a flat terrain(Z = 0 everywhere)

Repetdely adjusts that terrain to match obserevd brightness

Then it returns `DEM` and `albedo`

The Complete Explanation as follows It is to know that every part here is in sequence for this fnc

```python
def lambertian_reflectance(image, light_vector, max_iterations=100, lambda_smooth=0.1):
```

image: a 2D array of brightness values (gray image)

light_vector: (Lx, Ly, Lz) from earlier step

max_iterations: how many times to update the terrain

lambda_smooth: controls smoothness (lower = bumpier, higher = flatter)

```python
Lx, Ly, Lz = light_vector
```

get light vectors for x, y and z

```python
rows, cols = image.shape
I = np.maximum(image / np.nanmax(image), 1e-6)
Z = np.zeros_like(I)
albedo = np.ones_like(I)
```

Normalise intensities

Initialise DEM(Z) with 0 as initial heights

Initialse Reflectivity of 1 everywhere

```python
for _ in range(max_iterations):
    Z_new = np.copy(Z)
```

This loop runs many times to slowly improve the height map.

At each step, it updates Z to better match brightness.

```python
for i in range(1, rows-1):
    for j in range(1, cols-1):
```

Going Pixel by Pixel We say(skip border ones)

```
if np.isnan(I[i, j]):
    continue
```

If pixel Shadowed, Then Skip

```python
p = (Z[i+1, j] - Z[i, j])  # X direction slope
q = (Z[i, j+1] - Z[i, j])  # Y direction slope
```

Estimate Surfaces Slope

These represent terrain slope:

p = east-west (left-right slope)

q = north-south (up-down slope)

This is finite difference:

Means: How much elevation changes across one pixel.

### Computing Lambertian Reflectance

```python
denom = np.sqrt(p**2 + q**2 + 1)
R = (Lx * (-p) + Ly * (-q) + Lz) / denom
```

Calculate Reflectance

```python
albedo[i, j] = I[i, j] / R
albedo[i, j] = np.clip(albedo[i, j], 0.1, 1.0)
```

Estimate Albdeo through Reflectance just calculated

And Also clip so it stays between 0.1 to 1

```python
error = I[i, j] - albedo[i, j] * R
```

This is the difference between actual brightness and what we predicted

If error is positive, surface might be tilted wrong way (too dark)

If error is negative, too brigh

```python
dR_dZ = -(Lx + Ly) / denom
if abs(dR_dZ) > 1e-6:
    Z_new[i, j] = Z[i, j] + (error / dR_dZ)
```

Update Z Value Based on Error

This is the Jacobi iteration step:

It adjusts the height Z to make brightness prediction more accurate

`dR_dZ` = estimate of how sensitive R is to changes in Z

```python
Z_new[i, j] += lambda_smooth * (
    Z[i+1, j] + Z[i-1, j] + Z[i, j+1] + Z[i, j-1] - 4 * Z[i, j]
)
```

Add Smoothing effect. We dont want surface to be too uneven.

```python
Z = Z_new
Z[np.isnan(I)] = np.nanmean(Z)
```
Update the main DEM to the new version

Set all NaN areas (shadows) to the mean elevation, to keep it smooth

```python
return Z, albedo
```

the we simply return DEM and Albdeo

# Saving as GeoTiff

```python
def save_as_geotiff(data, output_path, lat=None, lon=None, pixel_resolution=18.06):
```

data: DEM and albdeo here

output_path: where to save as .tif

lat, lon: optional arrays of latitude and longitude

# Note
1. Albdeo estimation to be done much better
2. relative to absolute DIF?? To vague
