from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
import logging
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
    """Converts an equirectangular HDR image to a cross layout LDR cubemap with advanced anti-aliasing options"""

    # Input fields for basic parameters
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

    def generate_mipmap_pyramid(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate a mipmap pyramid for the input image

        Each level is half the size of the previous one, created using area averaging
        for high-quality downscaling.
        """
        mipmaps = [image]
        current = image

        while min(current.shape[:2]) > 1:
            # Calculate new dimensions (half of previous)
            new_height = max(1, current.shape[0] // 2)
            new_width = max(1, current.shape[1] // 2)

            # Downscale using area averaging for better quality
            next_level = cv2.resize(
                current,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA
            )
            mipmaps.append(next_level)
            current = next_level

            logger.info(
                f"Generated mipmap level {len(mipmaps)}: {next_level.shape}")

        return mipmaps

    def calculate_sampling_density(self, equirect: np.ndarray) -> np.ndarray:
        """Calculate the required sampling density based on image content

        Uses edge detection to identify areas that need more samples to prevent aliasing.
        Returns a map of sampling densities between 0 and 1.
        """
        # Convert to grayscale if needed
        if len(equirect.shape) == 3:
            gray = cv2.cvtColor(
                (equirect * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (equirect * 255).astype(np.uint8)

        # Calculate gradients in both directions
        gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Combine gradients and normalize
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        normalized = gradient_magnitude / np.max(gradient_magnitude)

        # Smooth the density map
        density_map = cv2.GaussianBlur(normalized, (5, 5), 0)

        return density_map

    def sample_pixel_bilinear(self,
                              equirect: np.ndarray,
                              u: float,
                              v: float,
                              mipmaps: Optional[List[np.ndarray]] = None,
                              sampling_rate: Optional[float] = None) -> np.ndarray:
        """Sample a pixel using bilinear interpolation

        If mipmaps are provided, uses appropriate mip level based on sampling rate.
        """
        # Select appropriate mipmap level if available
        if mipmaps is not None and sampling_rate is not None:
            # Calculate which mip level to use based on sampling rate
            mip_level = int(max(0, min(len(mipmaps) - 1,
                                       np.log2(sampling_rate))))
            source = mipmaps[mip_level]
        else:
            source = equirect

        h, w = source.shape[:2]

        # Convert to pixel coordinates
        x = u * (w - 1)
        y = v * (h - 1)

        # Get integer coordinates
        x0 = int(np.floor(x)) % w
        x1 = (x0 + 1) % w
        y0 = int(np.floor(y)) % h
        y1 = (y0 + 1) % h

        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0

        # Get corner values
        v00 = source[y0, x0]
        v01 = source[y0, x1]
        v10 = source[y1, x0]
        v11 = source[y1, x1]

        # Handle multi-channel images
        if len(source.shape) > 2:
            wx = wx[..., np.newaxis]
            wy = wy[..., np.newaxis]

        # Perform bilinear interpolation
        return (v00 * (1 - wx) * (1 - wy) +
                v01 * wx * (1 - wy) +
                v10 * (1 - wx) * wy +
                v11 * wx * wy)

    def supersample_pixel(self,
                          equirect: np.ndarray,
                          u: float,
                          v: float,
                          samples: int,
                          mipmaps: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Sample multiple points within a pixel area and average them

        Uses a regular grid of sample points for consistent quality.
        """
        total = np.zeros_like(equirect[0, 0], dtype=np.float32)
        h, w = equirect.shape[:2]

        # Calculate step size for sub-pixel sampling
        step = 1.0 / samples

        # Generate sample points in a regular grid
        for i in range(samples):
            for j in range(samples):
                # Calculate offset from pixel center
                u_offset = u + (i - samples/2 + 0.5) * step / w
                v_offset = v + (j - samples/2 + 0.5) * step / h

                # Wrap around for boundary cases
                u_offset = u_offset % 1.0
                v_offset = v_offset % 1.0

                # Sample with bilinear interpolation
                sample = self.sample_pixel_bilinear(
                    equirect, u_offset, v_offset, mipmaps)
                total += sample

        return total / (samples * samples)

    def equirect_to_cubemap(self, equirect: np.ndarray, face_size: int) -> Dict[str, np.ndarray]:
        """Convert equirectangular image to 6 cubemap faces with anti-aliasing

        Supports multiple anti-aliasing techniques that can be combined:
        - Bilinear interpolation
        - Supersampling
        - Mipmap filtering
        - Adaptive sampling
        """
        faces = {}
        face_names = ['front', 'back', 'left', 'right', 'up', 'down']

        # Generate mipmaps if enabled
        mipmaps = None
        if self.use_mipmaps:
            logger.info("Generating mipmap pyramid...")
            mipmaps = self.generate_mipmap_pyramid(equirect)

        # Calculate sampling density map if adaptive sampling is enabled
        density_map = None
        if self.use_adaptive:
            logger.info("Calculating sampling density map...")
            density_map = self.calculate_sampling_density(equirect)

        # Process each face
        for face_name in face_names:
            logger.info(f"Processing face: {face_name}")

            # Create face pixel coordinate grid
            xs, ys = np.meshgrid(
                np.linspace(-1, 1, face_size),
                np.linspace(-1, 1, face_size)
            )

            # # Convert face coordinates to 3D vectors
            # if face_name == 'front':
            #     x, y, z = xs, ys, np.ones_like(xs)
            # elif face_name == 'back':
            #     x, y, z = -xs, ys, -np.ones_like(xs)
            # elif face_name == 'left':
            #     x, y, z = -np.ones_like(xs), ys, xs
            # elif face_name == 'right':
            #     x, y, z = np.ones_like(xs), ys, -xs
            # elif face_name == 'up':
            #     x, y, z = -xs, -np.ones_like(xs), ys
            # elif face_name == 'down':
            #     x, y, z = -xs, np.ones_like(xs), -ys

            # # Convert face coordinates to 3D vectors
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

            # Initialize face array
            face = np.zeros((face_size, face_size, equirect.shape[2]),
                            dtype=np.float32)

            # Process each pixel in the face
            for i in range(face_size):
                for j in range(face_size):
                    # Get base sampling coordinates
                    u_coord = u[i, j]
                    v_coord = v[i, j]

                    # Calculate local sampling rate for mipmapping
                    if self.use_mipmaps or self.use_adaptive:
                        du_dx = np.abs(u[i, min(j+1, face_size-1)] -
                                       u[i, max(j-1, 0)]) * equirect.shape[1]
                        dv_dy = np.abs(v[min(i+1, face_size-1), j] -
                                       v[max(i-1, 0), j]) * equirect.shape[0]
                        sampling_rate = max(du_dx, dv_dy)

                        # Adjust sampling rate based on density map if adaptive
                        if self.use_adaptive:
                            density = density_map[
                                int(v_coord * density_map.shape[0]),
                                int(u_coord * density_map.shape[1])
                            ]
                            sampling_rate *= (1 + density)
                    else:
                        sampling_rate = None

                    # Sample the pixel
                    if self.use_supersampling:
                        face[i, j] = self.supersample_pixel(
                            equirect, u_coord, v_coord,
                            self.supersampling_rate, mipmaps
                        )
                    elif self.use_bilinear:
                        face[i, j] = self.sample_pixel_bilinear(
                            equirect, u_coord, v_coord,
                            mipmaps, sampling_rate
                        )
                    else:
                        # Fallback to nearest neighbor
                        y_idx = int(
                            v_coord * equirect.shape[0]) % equirect.shape[0]
                        x_idx = int(
                            u_coord * equirect.shape[1]) % equirect.shape[1]
                        face[i, j] = equirect[y_idx, x_idx]

            faces[face_name] = face

        return faces

    def create_cross_layout(self, faces: Dict[str, np.ndarray], face_size: int) -> np.ndarray:
        """Arrange faces in a cross layout

        Layout:
            U
          L F R B
            D
        """
        # Create blank image for the cross (3x4 faces)
        height = face_size * 3
        width = face_size * 4
        cross = np.zeros(
            (height, width, faces['front'].shape[2]),
            dtype=faces['front'].dtype
        )

        # Place faces in cross layout
        cross[0:face_size, face_size:face_size*2] = faces['up']
        cross[face_size:face_size*2, 0:face_size] = faces['left']
        cross[face_size:face_size*2, face_size:face_size*2] = faces['front']
        cross[face_size:face_size*2, face_size*2:face_size*3] = faces['right']
        cross[face_size:face_size*2, face_size*3:face_size*4] = faces['back']
        cross[face_size*2:face_size*3, face_size:face_size*2] = faces['down']

        return cross

    def invoke(self, context: InvocationContext) -> CubemapCrossOutput:
        """Process the equirectangular image and return cross layout cubemap

        The processing pipeline includes multiple anti-aliasing techniques that
        can be enabled/disabled through input parameters.
        """
        logger.info("Starting equirectangular to cubemap cross conversion...")
        logger.info(f"Anti-aliasing options:")
        logger.info(f"- Bilinear: {self.use_bilinear}")
        logger.info(f"- Supersampling: {self.use_supersampling}")
        logger.info(f"- Supersampling rate: {self.supersampling_rate}x")
        logger.info(f"- Mipmaps: {self.use_mipmaps}")
        logger.info(f"- Adaptive sampling: {self.use_adaptive}")

        # Load and validate input image
        logger.info(f"Loading input image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        logger.info(f"Loaded PIL image size: {image.size}, mode: {image.mode}")

        # Convert PIL to numpy float32 array (0.0 to 1.0 range)
        equirect = np.array(image).astype(np.float32) / 255.0

        # Convert to cubemap faces with anti-aliasing
        faces = self.equirect_to_cubemap(equirect, self.face_size)

        # Skip HDR to LDR conversion for now, just scale to 8-bit
        scaled_faces = {}
        for face_name, face in faces.items():
            # Clip values to valid range and convert to 8-bit
            scaled_faces[face_name] = np.clip(
                face * 255, 0, 255).astype(np.uint8)
            logger.info(f"Processed {face_name} face: shape={scaled_faces[face_name].shape}, "
                        f"dtype={scaled_faces[face_name].dtype}")

        # Create cross layout from processed faces
        cross = self.create_cross_layout(scaled_faces, self.face_size)
        logger.info(
            f"Created cross layout: shape={cross.shape}, dtype={cross.dtype}")

        # Convert to PIL and save
        pil_cross = Image.fromarray(cross)
        logger.info(
            f"Created PIL image: size={pil_cross.size}, mode={pil_cross.mode}")

        # Save the result
        image_dto = context.images.save(image=pil_cross)
        logger.info(
            f"Saved cross layout with image_name: {image_dto.image_name}")

        return CubemapCrossOutput(
            image=ImageField(image_name=image_dto.image_name)
        )
