from typing import Dict, Tuple
import numpy as np
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor
import cv2
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
            version="1.2.0")
class CubemapCrossToEquirectangularInvocation(BaseInvocation):
    """Converts a cross layout cubemap to an equirectangular image with optimized antialiasing"""

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

    # Performance options
    use_gpu: bool = InputField(
        default=True,
        description="Use GPU acceleration if available")
    num_threads: int = InputField(
        default=4, ge=1, le=16,
        description="Number of parallel processing threads")

    def check_gpu_availability(self) -> bool:
        """Check if CUDA GPU acceleration is available"""
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                return True
        except Exception:
            pass
        return False

    def extract_faces(self, cross_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract individual faces from cross layout using optimized array slicing"""
        height, width = cross_image.shape[:2]
        face_size = height // 3

        # Pre-calculate all slices for better performance
        faces = {
            'up': cross_image[0:face_size, face_size:face_size*2].copy(),
            'left': cross_image[face_size:face_size*2, 0:face_size].copy(),
            'front': cross_image[face_size:face_size*2, face_size:face_size*2].copy(),
            'right': cross_image[face_size:face_size*2, face_size*2:face_size*3].copy(),
            'back': cross_image[face_size:face_size*2, face_size*3:face_size*4].copy(),
            'down': cross_image[face_size*2:face_size*3, face_size:face_size*2].copy()
        }
        return faces

    def prepare_sampling_coordinates(self, height: int, width: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-calculate all sampling coordinates"""
        # Generate base coordinates
        phi = np.linspace(0, np.pi, height, endpoint=False)
        theta = np.linspace(0, 2*np.pi, width, endpoint=False)

        # Create meshgrid and convert to Cartesian coordinates
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Convert to Cartesian coordinates (vectorized)
        sin_phi = np.sin(phi_grid)
        x = sin_phi * np.cos(theta_grid)
        y = np.cos(phi_grid)
        z = sin_phi * np.sin(theta_grid)

        return x, y, z

    def prepare_face_coordinates(self, faces: Dict[str, np.ndarray], x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-calculate face selection and UV coordinates for all pixels"""
        # Calculate absolute values once
        abs_x = np.abs(x)
        abs_y = np.abs(y)
        abs_z = np.abs(z)

        # Initialize output arrays
        height, width = x.shape
        face_indices = np.zeros((height, width), dtype=np.int32)
        u_coords = np.zeros((height, width), dtype=np.float32)
        v_coords = np.zeros((height, width), dtype=np.float32)

        # Create masks for each face (vectorized operations)
        right_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x > 0)
        left_mask = (abs_x >= abs_y) & (abs_x >= abs_z) & (x <= 0)
        up_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y <= 0)
        down_mask = (abs_y >= abs_x) & (abs_y >= abs_z) & (y > 0)
        front_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z > 0)
        back_mask = (abs_z >= abs_x) & (abs_z >= abs_y) & (z <= 0)

        # Calculate UV coordinates for each face (vectorized)
        # Right face (index 0)
        face_indices[right_mask] = 0
        u_coords[right_mask] = -z[right_mask] / abs_x[right_mask]
        v_coords[right_mask] = y[right_mask] / abs_x[right_mask]

        # Left face (index 1)
        face_indices[left_mask] = 1
        u_coords[left_mask] = z[left_mask] / abs_x[left_mask]
        v_coords[left_mask] = y[left_mask] / abs_x[left_mask]

        # Up face (index 2)
        face_indices[up_mask] = 2
        u_coords[up_mask] = x[up_mask] / abs_y[up_mask]
        v_coords[up_mask] = z[up_mask] / abs_y[up_mask]

        # Down face (index 3)
        face_indices[down_mask] = 3
        u_coords[down_mask] = x[down_mask] / abs_y[down_mask]
        v_coords[down_mask] = -z[down_mask] / abs_y[down_mask]

        # Front face (index 4)
        face_indices[front_mask] = 4
        u_coords[front_mask] = x[front_mask] / abs_z[front_mask]
        v_coords[front_mask] = y[front_mask] / abs_z[front_mask]

        # Back face (index 5)
        face_indices[back_mask] = 5
        u_coords[back_mask] = -x[back_mask] / abs_z[back_mask]
        v_coords[back_mask] = y[back_mask] / abs_z[back_mask]

        return face_indices, u_coords, v_coords

    def process_chunk_gpu(self, faces: Dict[str, np.ndarray],
                          face_indices: np.ndarray, u_coords: np.ndarray,
                          v_coords: np.ndarray) -> np.ndarray:
        """Process a chunk of pixels using GPU acceleration"""
        face_list = [faces[name]
                     for name in ['right', 'left', 'up', 'down', 'front', 'back']]
        height, width = face_indices.shape
        channels = face_list[0].shape[2]
        result = np.zeros((height, width, channels), dtype=np.float32)

        # Convert coordinates to pixel space
        face_size = face_list[0].shape[0]
        u_pixels = ((u_coords + 1) / 2) * (face_size - 1)
        v_pixels = ((v_coords + 1) / 2) * (face_size - 1)

        # Process each face type separately using GPU
        for face_idx in range(6):
            mask = (face_indices == face_idx)
            if not np.any(mask):
                continue

            # Create map matrices for this face
            map_x = np.zeros(mask.shape, dtype=np.float32)
            map_y = np.zeros(mask.shape, dtype=np.float32)
            map_x[mask] = u_pixels[mask]
            map_y[mask] = v_pixels[mask]

            # Convert to GPU matrices
            gpu_face = cv2.cuda_GpuMat(face_list[face_idx])
            gpu_map_x = cv2.cuda_GpuMat(map_x)
            gpu_map_y = cv2.cuda_GpuMat(map_y)

            # Remap on GPU
            gpu_result = cv2.cuda.remap(gpu_face, gpu_map_x, gpu_map_y,
                                        interpolation=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_WRAP)

            # Copy back to CPU
            cpu_result = gpu_result.download()
            result[mask] = cpu_result[mask]

        return result

    def process_chunk_cpu(self, faces: Dict[str, np.ndarray],
                          face_indices: np.ndarray, u_coords: np.ndarray,
                          v_coords: np.ndarray) -> np.ndarray:
        """Process a chunk of pixels using CPU with vectorized operations"""
        face_list = [faces[name]
                     for name in ['right', 'left', 'up', 'down', 'front', 'back']]
        height, width = face_indices.shape
        channels = face_list[0].shape[2]
        result = np.zeros((height, width, channels), dtype=np.float32)

        # Convert coordinates to pixel space
        face_size = face_list[0].shape[0]
        u_pixels = ((u_coords + 1) / 2) * (face_size - 1)
        v_pixels = ((v_coords + 1) / 2) * (face_size - 1)

        # Process each face type
        for face_idx in range(6):
            mask = (face_indices == face_idx)
            if not np.any(mask):
                continue

            # Use OpenCV's remap function for efficient interpolation
            map_x = np.zeros(mask.shape, dtype=np.float32)
            map_y = np.zeros(mask.shape, dtype=np.float32)
            map_x[mask] = u_pixels[mask]
            map_y[mask] = v_pixels[mask]

            remapped = cv2.remap(face_list[face_idx], map_x, map_y,
                                 interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP)

            result[mask] = remapped[mask]

        return result

    def cubemap_to_equirect(self, faces: Dict[str, np.ndarray],
                            width: int, height: int) -> np.ndarray:
        """Convert cubemap faces to equirectangular image using optimized processing"""
        logger.info("Starting optimized equirectangular conversion...")

        # Pre-calculate all coordinates
        x, y, z = self.prepare_sampling_coordinates(height, width)
        face_indices, u_coords, v_coords = self.prepare_face_coordinates(
            faces, x, y, z)

        # Determine processing method
        use_gpu = self.use_gpu and self.check_gpu_availability()
        logger.info(f"Using GPU acceleration: {use_gpu}")

        if use_gpu:
            # Process entire image at once on GPU
            result = self.process_chunk_gpu(
                faces, face_indices, u_coords, v_coords)
        else:
            # Split into chunks for parallel CPU processing
            chunk_size = height // self.num_threads
            chunks = []

            for i in range(0, height, chunk_size):
                end = min(i + chunk_size, height)
                chunks.append((
                    faces,
                    face_indices[i:end],
                    u_coords[i:end],
                    v_coords[i:end]
                ))

            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                chunk_results = list(executor.map(
                    lambda x: self.process_chunk_cpu(*x), chunks))

            # Combine results
            result = np.concatenate(chunk_results, axis=0)

        return result

    def invoke(self, context: InvocationContext) -> EquirectangularOutput:
        """Process the cubemap cross image and return equirectangular image"""
        # Load and prepare input image
        logger.info(f"Loading input image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        cross = np.array(image).astype(np.float32) / 255.0

        # Extract faces and convert to equirectangular
        faces = self.extract_faces(cross)
        equirect = self.cubemap_to_equirect(
            faces, self.output_width, self.output_height)

        # Post-process and save
        equirect_8bit = np.clip(equirect * 255, 0, 255).astype(np.uint8)
        pil_equirect = Image.fromarray(equirect_8bit)
        pil_equirect = pil_equirect.transpose(Image.FLIP_TOP_BOTTOM)
        pil_equirect = pil_equirect.transpose(Image.FLIP_LEFT_RIGHT)

        # Save and return
        image_dto = context.images.save(image=pil_equirect)
        return EquirectangularOutput(
            image=ImageField(image_name=image_dto.image_name)
        )
