from typing import Dict
import numpy as np
from PIL import Image
import logging
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output, InvocationContext
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.primitives import ImageField

logger = logging.getLogger("InvokeAI")


@invocation_output("cubemap_output")
class CubemapOutput(BaseInvocationOutput):
    """Output type for cubemap faces"""
    front: ImageField = OutputField(description="Front face of cubemap")
    back: ImageField = OutputField(description="Back face of cubemap")
    left: ImageField = OutputField(description="Left face of cubemap")
    right: ImageField = OutputField(description="Right face of cubemap")
    up: ImageField = OutputField(description="Up face of cubemap")
    down: ImageField = OutputField(description="Down face of cubemap")


@invocation("equirect_to_cubemap",
            title="Equirectangular to Cubemap",
            tags=["image", "panorama", "cubemap", "equirectangular", "hdr"],
            category="Image/Panorama",
            version="1.0.0")
class EquirectangularToCubemapInvocation(BaseInvocation):
    """Converts an equirectangular HDR image to 6 separate LDR cubemap faces"""

    # Input fields
    image: ImageField = InputField(
        description="Input equirectangular HDR image")
    face_size: int = InputField(
        default=1024, ge=64, le=4096, description="Size of each cubemap face")
    exposure: float = InputField(
        default=0.0, ge=-10.0, le=10.0, description="Exposure adjustment for tone mapping")

    def equirect_to_cubemap(self, equirect: np.ndarray, face_size: int) -> Dict[str, np.ndarray]:
        """Convert equirectangular image to 6 cubemap faces"""

        logger.info(
            f"Input equirect shape: {equirect.shape}, dtype: {equirect.dtype}")
        logger.info(
            f"Input value range: min={np.min(equirect)}, max={np.max(equirect)}")

        faces = {}
        face_names = ['front', 'back', 'left', 'right', 'up', 'down']

        for face_name in face_names:
            # Create meshgrid for the face
            xs, ys = np.meshgrid(np.linspace(-1, 1, face_size),
                                 np.linspace(-1, 1, face_size))

            # Convert face coordinates to 3D vectors
            if face_name == 'front':
                x, y, z = xs, ys, np.ones_like(xs)
            elif face_name == 'back':
                x, y, z = -xs, ys, -np.ones_like(xs)
            elif face_name == 'left':
                x, y, z = -np.ones_like(xs), ys, xs
            elif face_name == 'right':
                x, y, z = np.ones_like(xs), ys, -xs
            elif face_name == 'up':
                x, y, z = xs, -np.ones_like(xs), ys
            elif face_name == 'down':
                x, y, z = xs, np.ones_like(xs), -ys

            # Convert to spherical coordinates
            theta = np.arctan2(z, x)
            phi = np.arctan2(y, np.sqrt(x**2 + z**2))

            # Map to equirectangular coordinates
            u = (theta + np.pi) / (2 * np.pi)
            v = (phi + np.pi/2) / np.pi

            # Sample from equirectangular image
            h, w = equirect.shape[:2]
            x_idx = (u * w).astype(int) % w
            y_idx = (v * h).astype(int) % h

            faces[face_name] = equirect[y_idx, x_idx]

            logger.info(
                f"{face_name} face shape: {faces[face_name].shape}, dtype: {faces[face_name].dtype}")
            logger.info(
                f"{face_name} face range: min={np.min(faces[face_name])}, max={np.max(faces[face_name])}")

        return faces

    def hdr_to_ldr(self, hdr_image: np.ndarray, exposure: float = 0.0) -> np.ndarray:
        """Convert HDR image to LDR using Reinhard tone mapping"""

        logger.info(
            f"HDR input shape: {hdr_image.shape}, dtype: {hdr_image.dtype}")
        logger.info(
            f"HDR value range: min={np.min(hdr_image)}, max={np.max(hdr_image)}")

        # Apply exposure adjustment
        adjusted = hdr_image * (2.0 ** exposure)

        # Reinhard tone mapping
        L = 0.2126 * adjusted[..., 0] + 0.7152 * \
            adjusted[..., 1] + 0.0722 * adjusted[..., 2]
        L_white = np.max(L)
        L_scaled = L * (1 + L/L_white**2) / (1 + L)

        # Scale colors
        ratio = L_scaled / (L + 1e-8)
        ldr = adjusted * ratio[..., None]

        # Convert to 8-bit
        result = np.clip(ldr * 255, 0, 255).astype(np.uint8)

        logger.info(f"LDR output shape: {result.shape}, dtype: {result.dtype}")
        logger.info(
            f"LDR value range: min={np.min(result)}, max={np.max(result)}")

        return result

    def invoke(self, context: InvocationContext) -> CubemapOutput:
        """Process the equirectangular image and return 6 cubemap faces"""

        logger.info(f"Loading input image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        logger.info(f"Loaded PIL image size: {image.size}, mode: {image.mode}")

        # Convert PIL to numpy
        equirect = np.array(image).astype(np.float32) / 255.0

        # Convert to cubemap faces
        faces = self.equirect_to_cubemap(equirect, self.face_size)

        # Convert each face from HDR to LDR and save
        outputs = {}
        for face_name, face in faces.items():
            # Apply tone mapping
            ldr_face = self.hdr_to_ldr(face, self.exposure)

            # Convert to PIL image
            pil_face = Image.fromarray(ldr_face)
            logger.info(
                f"Created PIL face {face_name}: size={pil_face.size}, mode={pil_face.mode}")

            # Save the face
            image_dto = context.images.save(image=pil_face)
            logger.info(
                f"Saved face {face_name} with image_name: {image_dto.image_name}")

            # Create ImageField from the saved image name
            image_field = ImageField(image_name=image_dto.image_name)
            logger.info(f"Created ImageField for {face_name}: {image_field}")
            outputs[face_name] = image_field

        # Return using our custom output type
        try:
            result = CubemapOutput(**outputs)
            logger.info("Successfully created CubemapOutput")
            return result
        except Exception as e:
            logger.error(f"Failed to create CubemapOutput: {str(e)}")
            for face_name, output in outputs.items():
                logger.error(f"{face_name} output type: {type(output)}")
            raise
