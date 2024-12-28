# Panorama Processing for InvokeAI

This package provides a comprehensive set of tools for processing panoramic images within the InvokeAI framework. It enables conversion between equirectangular, cubemap, and cross-layout formats, with support for both HDR and LDR image processing.

## Features

The package includes several key components for panorama processing:

### Format Conversion
- Equirectangular to Cubemap (individual faces)
- Equirectangular to Cross-layout Cubemap
- Cross-layout Cubemap to Equirectangular
- LDR to HDR conversion with OpenEXR support

### Advanced Processing Options
- Multiple anti-aliasing methods:
  - Bilinear interpolation
  - Supersampling (up to 16x)
  - Mipmap filtering
  - Adaptive sampling based on image content
- HDR processing with exposure control
- GPU acceleration using CUDA
- Multi-threaded CPU processing
- Intel IPP optimizations

## Installation

The package is designed to be used within the InvokeAI framework. Ensure you have the following dependencies installed:

```bash
pip install numpy pillow opencv-python openexr
```

For GPU acceleration, ensure you have CUDA support installed with OpenCV.

## Usage

### Basic Conversion Examples

1. Converting Equirectangular to Cubemap:

```python
from invokeai.app.invocations.panorama import EquirectangularToCubemapInvocation

invocation = EquirectangularToCubemapInvocation(
    image=your_image,
    face_size=1024,
    exposure=0.0
)
result = invocation.invoke(context)
```

2. Converting Equirectangular to Cross Layout:

```python
from invokeai.app.invocations.panorama import EquirectangularToCubemapCrossInvocation

invocation = EquirectangularToCubemapCrossInvocation(
    image=your_image,
    face_size=1024,
    exposure=0.0,
    use_bilinear=True,
    use_supersampling=True,
    supersampling_rate=2
)
result = invocation.invoke(context)
```

3. Converting Cross Layout back to Equirectangular:

```python
from invokeai.app.invocations.panorama import CubemapCrossToEquirectangularInvocation

invocation = CubemapCrossToEquirectangularInvocation(
    image=your_image,
    output_width=4096,
    output_height=2048
)
result = invocation.invoke(context)
```

4. Converting LDR to HDR:

```python
from invokeai.app.invocations.panorama import LDRtoHDRInvocation

invocation = LDRtoHDRInvocation(
    image=your_image,
    output_path="output.exr",
    gamma=2.2,
    brightness=1.0
)
result = invocation.invoke(context)
```

### Advanced Usage with GPU Acceleration

For optimal performance with large images, you can enable GPU acceleration and customize processing options:

```python
invocation = EquirectangularToCubemapCrossInvocation(
    image=your_image,
    face_size=2048,
    use_gpu=True,
    use_ipp=True,
    num_threads=8,
    batch_size=64,
    use_supersampling=True,
    supersampling_rate=2,
    use_adaptive=True
)
```

## Technical Details

### Coordinate Systems

The package handles several coordinate systems:

1. Equirectangular (spherical coordinates):
   - Latitude: -π/2 to π/2
   - Longitude: -π to π

2. Cubemap faces (normalized coordinates):
   - Each face uses [-1, 1] × [-1, 1] coordinate space
   - Face orientations follow OpenGL conventions

3. Cross layout:
   ```
       [U]
   [L][F][R][B]
       [D]
   ```
   Where: F=Front, B=Back, L=Left, R=Right, U=Up, D=Down

### Anti-aliasing Methods

The package implements several anti-aliasing techniques:

1. Bilinear Interpolation
   - Smooth sampling between pixels
   - Good balance of quality and performance

2. Supersampling
   - Multiple samples per pixel
   - Rates: 2x2, 3x3, or 4x4 samples
   - Higher quality but more computationally intensive

3. Mipmap Filtering
   - Pre-filtered image pyramids
   - Automatic level selection based on sampling rate
   - Reduces aliasing for minification

4. Adaptive Sampling
   - Analyzes image content for edge detection
   - Increases sample density in high-frequency areas
   - Optimizes processing resources

### Performance Optimizations

The codebase includes several optimization strategies:

1. GPU Acceleration
   - CUDA support through OpenCV
   - Parallel processing of image batches
   - Efficient memory management

2. CPU Optimizations
   - Multi-threaded processing
   - Intel IPP integration
   - Vectorized operations using NumPy
   - Efficient array slicing and batch processing

3. Memory Management
   - Streaming large images
   - Batch processing for GPU operations
   - Efficient coordinate pre-calculation

## Error Handling

The package includes comprehensive error handling for:
- Input validation
- File I/O operations
- GPU memory management
- Resource cleanup
- Graceful fallbacks for unavailable optimizations

## Contributing

When contributing to this codebase:

1. Follow the existing code style and documentation patterns
2. Add unit tests for new functionality
3. Update documentation for API changes
4. Test both CPU and GPU paths
5. Verify memory handling for large images

## License

This package is part of the InvokeAI project and follows its licensing terms.

## Acknowledgments

This implementation builds on established panoramic processing techniques and optimizes them for modern hardware and the InvokeAI framework.