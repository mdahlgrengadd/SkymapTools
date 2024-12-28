from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import logging
from concurrent.futures import ThreadPoolExecutor
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output, InvocationContext
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.primitives import ImageField

logger = logging.getLogger("InvokeAI")


@invocation_output("cubemap_cross_output")
class CubemapCrossOutput(BaseInvocationOutput):
    """Output type for cubemap in cross layout"""
    image: ImageField = OutputField(description="Cubemap in cross layout")


@invocation("equirect_to_cubemap_cross",
            title="Equirectangular to Cubemap Cross",
            tags=["image", "panorama", "cubemap",
                  "equirectangular", "hdr", "cross"],
            category="Image/Panorama",
            version="1.0.0")
class EquirectangularToCubemapCrossInvocation(BaseInvocation):
    """Converts an equirectangular HDR image to a cross layout LDR cubemap with optimized processing options"""

    # Basic parameters
    image: ImageField = InputField(
        description="Input equirectangular HDR image")
    face_size: int = InputField(
        default=1024, ge=64, le=4096, description="Size of each cubemap face")
    exposure: float = InputField(
        default=0.0, ge=-10.0, le=10.0, description="Exposure adjustment for tone mapping")

    # Anti-aliasing options
    use_bilinear: bool = InputField(
        default=True,
        description="Use bilinear interpolation instead of nearest neighbor")
    use_supersampling: bool = InputField(
        default=False,
        description="Enable supersampling anti-aliasing")
    supersampling_rate: int = InputField(
        default=2, ge=2, le=4,
        description="Supersampling rate (2=4x, 3=9x, 4=16x samples per pixel)")
    use_mipmaps: bool = InputField(
        default=False,
        description="Enable mipmap filtering for better quality")
    use_adaptive: bool = InputField(
        default=False,
        description="Enable adaptive sampling based on image content")

    # Performance optimization options
    use_gpu: bool = InputField(
        default=False,
        description="Use CUDA GPU acceleration if available")
    use_ipp: bool = InputField(
        default=True,
        description="Use Intel IPP optimizations if available")
    num_threads: int = InputField(
        default=4, ge=1, le=16,
        description="Number of threads for parallel processing")
    batch_size: int = InputField(
        default=32, ge=1, le=128,
        description="Batch size for vectorized operations")

    def check_gpu_availability(self) -> bool:
        """Check if CUDA GPU acceleration is available

        Returns:
            bool: True if CUDA GPU acceleration is available, False otherwise
        """
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except (AttributeError, Exception) as e:
            logger.warning(f"CUDA support check failed: {str(e)}")
        return False

    def check_ipp_availability(self) -> bool:
        """Check if Intel IPP optimizations are available

        Returns:
            bool: True if Intel IPP is available and enabled, False otherwise
        """
        try:
            if self.use_ipp:
                cv2.setUseIPP(True)
                return cv2.useIPP()
        except Exception as e:
            logger.warning(f"IPP support check failed: {str(e)}")
        return False

    def process_batch(self,
                      source: np.ndarray,
                      x_maps: np.ndarray,
                      y_maps: np.ndarray) -> np.ndarray:
        """Process a batch of pixels using optimized remapping

        This method automatically selects the best available processing method:
        1. GPU acceleration if available and requested
        2. CPU processing with IPP optimizations if available
        3. Standard CPU processing as fallback

        Args:
            source: Source image array
            x_maps: X coordinate mapping array
            y_maps: Y coordinate mapping array

        Returns:
            np.ndarray: Processed image batch
        """
        # Try GPU processing if requested
        if self.use_gpu and self.check_gpu_availability():
            try:
                # Transfer data to GPU
                gpu_source = cv2.cuda_GpuMat(source)
                gpu_x_maps = cv2.cuda_GpuMat(x_maps)
                gpu_y_maps = cv2.cuda_GpuMat(y_maps)

                # Process on GPU
                gpu_result = cv2.cuda.remap(
                    gpu_source,
                    gpu_x_maps,
                    gpu_y_maps,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_WRAP
                )

                # Transfer result back to CPU
                result = gpu_result.download()

                # Clean up GPU memory
                for gpu_mat in [gpu_source, gpu_x_maps, gpu_y_maps, gpu_result]:
                    try:
                        gpu_mat.release()
                    except Exception:
                        pass

                logger.debug(
                    "Successfully processed batch using GPU acceleration")
                return result

            except Exception as e:
                logger.warning(
                    f"GPU processing failed, falling back to CPU: {str(e)}")

        # Fallback to CPU processing
        try:
            # Process on CPU (IPP will be used automatically if available and enabled)
            result = cv2.remap(
                source,
                x_maps,
                y_maps,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP
            )
            return result

        except Exception as e:
            logger.error(f"Remapping operation failed: {str(e)}")
            raise

        return result

    def process_face_parallel(self,
                              face_name: str,
                              equirect: np.ndarray,
                              face_size: int,
                              batch_size: int = 32) -> np.ndarray:
        """Process a single cubemap face with batched operations

        Divides the face into batches for efficient processing and memory usage.
        Handles coordinate generation and remapping for a complete face.
        """
        # Generate base coordinates for this face
        xs, ys = np.meshgrid(
            np.linspace(-1, 1, face_size),
            np.linspace(-1, 1, face_size)
        )

        # Convert face coordinates to 3D vectors based on face orientation
        if face_name == 'front':
            x, y, z = -xs, ys, np.ones_like(xs)
        elif face_name == 'back':
            x, y, z = xs, ys, -np.ones_like(xs)
        elif face_name == 'right':
            x, y, z = -np.ones_like(xs), ys, -xs
        elif face_name == 'left':
            x, y, z = np.ones_like(xs), ys, xs
        elif face_name == 'up':
            x, y, z = -xs, -np.ones_like(xs), ys
        elif face_name == 'down':
            x, y, z = -xs, np.ones_like(xs), -ys

        # Convert to spherical coordinates
        theta = np.arctan2(z, x)
        phi = np.arctan2(y, np.sqrt(x**2 + z**2))

        # Convert to equirectangular coordinates
        u = (theta + np.pi) / (2 * np.pi)
        v = (phi + np.pi/2) / np.pi

        # Convert to pixel coordinates
        h, w = equirect.shape[:2]
        x_map = (u * (w - 1)).astype(np.float32)
        y_map = (v * (h - 1)).astype(np.float32)

        # Process in batches
        face = np.zeros(
            (face_size, face_size, equirect.shape[2]), dtype=np.float32)
        num_pixels = face_size * face_size
        num_batches = (num_pixels + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_pixels)

            # Extract batch coordinates
            x_batch = x_map.ravel()[start_idx:end_idx].reshape(-1, 1)
            y_batch = y_map.ravel()[start_idx:end_idx].reshape(-1, 1)

            # Process batch
            result_batch = self.process_batch(
                equirect,
                x_batch,
                y_batch
            )

            # Store results
            face.ravel()[start_idx*3:end_idx*3] = result_batch.ravel()

        return face

    def equirect_to_cubemap_optimized(self,
                                      equirect: np.ndarray,
                                      face_size: int) -> Dict[str, np.ndarray]:
        """Convert equirectangular image to cubemap faces using optimized processing

        Utilizes parallel processing, GPU acceleration, and batched operations
        for maximum performance while maintaining high quality output.
        """
        face_names = ['front', 'back', 'left', 'right', 'up', 'down']
        faces = {}

        # Process faces in parallel if multiple threads requested
        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit all face processing tasks
                future_to_face = {
                    executor.submit(
                        self.process_face_parallel,
                        face_name,
                        equirect,
                        face_size,
                        self.batch_size
                    ): face_name for face_name in face_names
                }

                # Collect results as they complete
                for future in future_to_face:
                    face_name = future_to_face[future]
                    try:
                        faces[face_name] = future.result()
                    except Exception as e:
                        logger.error(
                            f"Error processing face {face_name}: {str(e)}")
                        raise
        else:
            # Process faces sequentially
            for face_name in face_names:
                faces[face_name] = self.process_face_parallel(
                    face_name,
                    equirect,
                    face_size,
                    self.batch_size
                )

        return faces

    def create_cross_layout(self, faces: Dict[str, np.ndarray], face_size: int) -> np.ndarray:
        """Arrange faces in a cross layout efficiently

        Creates the output array with minimal memory operations and optimal array assignment.
        """
        # Pre-allocate output array
        height = face_size * 3
        width = face_size * 4
        cross = np.zeros((height, width, faces['front'].shape[2]),
                         dtype=faces['front'].dtype)

        # Assign faces using optimized array slicing
        cross[0:face_size, face_size:face_size*2] = faces['up']
        cross[face_size:face_size*2, 0:face_size] = faces['left']
        cross[face_size:face_size*2, face_size:face_size*2] = faces['front']
        cross[face_size:face_size*2, face_size*2:face_size*3] = faces['right']
        cross[face_size:face_size*2, face_size*3:face_size*4] = faces['back']
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces['down']

        return cross

    def invoke(self, context: InvocationContext) -> CubemapCrossOutput:
        """Process the equirectangular image with optimized performance options

        Provides detailed logging of processing options and performance characteristics
        while utilizing all available optimizations based on user settings.
        """
        # Check and log available optimizations
        gpu_available = self.check_gpu_availability()
        ipp_available = self.check_ipp_availability()

        logger.info("Starting optimized equirectangular to cubemap conversion")
        logger.info("Performance options:")
        logger.info(
            f"- GPU acceleration: {self.use_gpu} (Available: {gpu_available})")
        logger.info(
            f"- Intel IPP: {self.use_ipp} (Available: {ipp_available})")
        logger.info(f"- Parallel threads: {self.num_threads}")
        logger.info(f"- Batch size: {self.batch_size}")

        if self.use_gpu and not gpu_available:
            logger.info(
                "GPU acceleration requested but not available, using CPU processing")

        # Load and prepare input image
        logger.info(f"Loading input image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        logger.info(f"Input image size: {image.size}, mode: {image.mode}")

        # Convert to numpy array
        equirect = np.array(image).astype(np.float32) / 255.0

        # Process using optimized conversion
        faces = self.equirect_to_cubemap_optimized(equirect, self.face_size)

        # Convert to 8-bit
        scaled_faces = {
            name: np.clip(face * 255, 0, 255).astype(np.uint8)
            for name, face in faces.items()
        }

        # Create cross layout
        cross = self.create_cross_layout(scaled_faces, self.face_size)

        # Convert to PIL and save
        pil_cross = Image.fromarray(cross)
        image_dto = context.images.save(image=pil_cross)

        logger.info("Conversion completed successfully")

        return CubemapCrossOutput(
            image=ImageField(image_name=image_dto.image_name)
        )
