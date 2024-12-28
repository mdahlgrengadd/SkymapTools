from typing import Dict, Tuple
import numpy as np
from PIL import Image
import logging
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output, InvocationContext
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.primitives import ImageField

logger = logging.getLogger("InvokeAI")


@invocation_output("equirect_output")
class EquirectangularOutput(BaseInvocationOutput):
    """Output type for equirectangular image"""
    image: ImageField = OutputField(
        description="Converted equirectangular image")


@invocation("cubemap_cross_to_equirect",
            title="Cubemap Cross to Equirectangular",
            tags=["image", "panorama", "cubemap",
                  "equirectangular", "hdr", "cross"],
            category="Image/Panorama",
            version="1.1.0")
class CubemapCrossToEquirectangularInvocation(BaseInvocation):
    """Converts a cross layout cubemap to an equirectangular image with antialiasing"""

    # Input fields
    image: ImageField = InputField(
        description="Input cubemap cross layout image")
    output_width: int = InputField(
        default=4096, ge=1024, le=8192,
        description="Width of output equirectangular image")
    output_height: int = InputField(
        default=2048, ge=512, le=4096,
        description="Height of output equirectangular image")

    # Antialiasing options
    use_bilinear: bool = InputField(
        default=True,
        description="Use bilinear interpolation for smoother sampling")
    use_supersampling: bool = InputField(
        default=False,
        description="Enable supersampling antialiasing")
    supersample_rate: int = InputField(
        default=2, ge=2, le=4,
        description="Supersampling rate (2=4x, 3=9x, 4=16x samples)")

    def extract_faces(self, cross_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual faces from cross layout"""
        height, width = cross_image.shape[:2]
        face_size = height // 3  # Since cross layout is 3 faces tall

        faces = {
            'up': cross_image[0:face_size, face_size:face_size*2],
            'left': cross_image[face_size:face_size*2, 0:face_size],
            'front': cross_image[face_size:face_size*2, face_size:face_size*2],
            'right': cross_image[face_size:face_size*2, face_size*2:face_size*3],
            'back': cross_image[face_size:face_size*2, face_size*3:face_size*4],
            'down': cross_image[face_size*2:face_size*3, face_size:face_size*2]
        }

        return faces

    def sample_face_bilinear(self, face: np.ndarray, u: float, v: float) -> np.ndarray:
        """Sample from a face using bilinear interpolation

        Args:
            face: Face image array
            u, v: Normalized coordinates (0 to 1)

        Returns:
            Interpolated pixel value
        """
        h, w = face.shape[:2]

        # Convert to pixel coordinates
        x = u * (w - 1)
        y = v * (h - 1)

        # Get integer coordinates
        x0 = int(np.floor(x))
        x1 = min(x0 + 1, w - 1)
        y0 = int(np.floor(y))
        y1 = min(y0 + 1, h - 1)

        # Calculate interpolation weights
        wx = x - x0
        wy = y - y0

        # Sample corners
        c00 = face[y0, x0]
        c01 = face[y0, x1]
        c10 = face[y1, x0]
        c11 = face[y1, x1]

        # Interpolate
        return (c00 * (1 - wx) * (1 - wy) +
                c01 * wx * (1 - wy) +
                c10 * (1 - wx) * wy +
                c11 * wx * wy)

    def get_face_coordinates(self, x: float, y: float, z: float) -> Tuple[str, float, float]:
        """Determine which face a 3D point lies on and its coordinates on that face"""
        abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)

        if abs_x >= abs_y and abs_x >= abs_z:
            # Right or Left face
            if x > 0:
                return 'right', -z/abs_x, y/abs_x
            else:
                return 'left', z/abs_x, y/abs_x
        elif abs_y >= abs_x and abs_y >= abs_z:
            # Up or Down face
            if y > 0:
                return 'down', x/abs_y, -z/abs_y
            else:
                return 'up', x/abs_y, z/abs_y
        else:
            # Front or Back face
            if z > 0:
                return 'front', x/abs_z, y/abs_z
            else:
                return 'back', -x/abs_z, y/abs_z

    def sample_direction(self, faces: Dict[str, np.ndarray],
                         direction: Tuple[float, float, float]) -> np.ndarray:
        """Sample from cubemap faces in a given direction with antialiasing

        Args:
            faces: Dictionary of cubemap faces
            direction: (x, y, z) direction vector

        Returns:
            Sampled pixel value
        """
        # Get face and coordinates
        face_name, u, v = self.get_face_coordinates(*direction)
        face = faces[face_name]

        # Convert from [-1,1] to [0,1] range
        u = (u + 1) / 2
        v = (v + 1) / 2

        if self.use_bilinear:
            return self.sample_face_bilinear(face, u, v)
        else:
            # Fallback to nearest neighbor
            h, w = face.shape[:2]
            x = min(int(u * w), w - 1)
            y = min(int(v * h), h - 1)
            return face[y, x]

    def cubemap_to_equirect(self, faces: Dict[str, np.ndarray],
                            width: int, height: int) -> np.ndarray:
        """Convert cubemap faces to equirectangular image with antialiasing"""
        equirect = np.zeros((height, width, faces['front'].shape[2]),
                            dtype=np.float32)

        # Generate spherical coordinates
        # Use height+1 and width+1 points and exclude the last one to avoid overlap
        phi = np.linspace(0, np.pi, height, endpoint=False)
        theta = np.linspace(0, 2*np.pi, width, endpoint=False)

        # Create meshgrid and convert to Cartesian coordinates
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        x = np.sin(phi_grid) * np.cos(theta_grid)
        y = np.cos(phi_grid)
        z = np.sin(phi_grid) * np.sin(theta_grid)

        logger.info("Starting equirectangular conversion with antialiasing...")
        logger.info(f"Antialiasing settings:")
        logger.info(f"- Bilinear: {self.use_bilinear}")
        logger.info(f"- Supersampling: {self.use_supersampling}")
        logger.info(f"- Supersample rate: {self.supersample_rate}")

        # Process each pixel
        for i in range(height):
            for j in range(width):
                if self.use_supersampling:
                    # Initialize accumulator
                    sample_sum = np.zeros_like(faces['front'][0, 0])
                    samples = self.supersample_rate ** 2

                    # Generate subpixel samples
                    for si in range(self.supersample_rate):
                        for sj in range(self.supersample_rate):
                            # Calculate subpixel offset
                            dp = (si + 0.5) / self.supersample_rate - 0.5
                            dt = (sj + 0.5) / self.supersample_rate - 0.5

                            # Calculate direction for this sample
                            phi_sample = phi[i, j] + dp * (np.pi / height)
                            theta_sample = theta[i, j] + dt * (2*np.pi / width)

                            x_sample = np.sin(phi_sample) * \
                                np.cos(theta_sample)
                            y_sample = np.cos(phi_sample)
                            z_sample = np.sin(phi_sample) * \
                                np.sin(theta_sample)

                            # Accumulate sample
                            sample_sum += self.sample_direction(
                                faces, (x_sample, y_sample, z_sample))

                    # Average samples
                    equirect[i, j] = sample_sum / samples
                else:
                    # Single sample per pixel
                    equirect[i, j] = self.sample_direction(
                        faces, (x[i, j], y[i, j], z[i, j]))

        return equirect

    def invoke(self, context: InvocationContext) -> EquirectangularOutput:
        """Process the cubemap cross image and return equirectangular image"""
        logger.info(f"Loading input image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        logger.info(f"Loaded PIL image size: {image.size}, mode: {image.mode}")

        # Convert PIL to numpy
        cross = np.array(image).astype(np.float32) / 255.0

        # Extract faces from cross layout
        faces = self.extract_faces(cross)

        # Convert to equirectangular with antialiasing
        equirect = self.cubemap_to_equirect(
            faces, self.output_width, self.output_height)

        # Convert back to 8-bit
        equirect_8bit = np.clip(equirect * 255, 0, 255).astype(np.uint8)

        # Convert to PIL and apply required transformations
        pil_equirect = Image.fromarray(equirect_8bit)
        pil_equirect = pil_equirect.transpose(Image.FLIP_TOP_BOTTOM)
        pil_equirect = pil_equirect.transpose(Image.FLIP_LEFT_RIGHT)

        logger.info(
            f"Created equirectangular image: size={pil_equirect.size}, mode={pil_equirect.mode}")

        # Save and return
        image_dto = context.images.save(image=pil_equirect)
        logger.info(
            f"Saved equirectangular image with image_name: {image_dto.image_name}")

        return EquirectangularOutput(
            image=ImageField(image_name=image_dto.image_name)
        )
