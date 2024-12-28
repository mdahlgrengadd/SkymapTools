from typing import Literal
import numpy as np
from PIL import Image
import logging
import OpenEXR
import cv2
import os
from pathlib import Path
from io import BytesIO
from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output, InvocationContext
from invokeai.app.invocations.fields import InputField, OutputField
from invokeai.app.invocations.primitives import ImageField

logger = logging.getLogger("InvokeAI")


@invocation_output("ldr_to_hdr_output")
class LDRtoHDROutput(BaseInvocationOutput):
    """Metadata about the saved HDR file"""
    file_path: str = OutputField(
        description="Path where the HDR file was saved")
    format: str = OutputField(
        description="Format of the saved file (exr or hdr)")
    file_size: int = OutputField(description="Size of the saved file in bytes")


@invocation("ldr_to_hdr",
            title="Convert LDR to HDR",
            tags=["image", "hdr", "openexr", "radiance", "conversion"],
            category="Image/Convert",
            version="1.0.0")
class LDRtoHDRInvocation(BaseInvocation):
    """Converts a standard LDR image to HDR format and saves to disk"""

    # Input fields
    image: ImageField = InputField(description="Input LDR image to convert")
    output_path: str = InputField(
        description="Full file path where to save the HDR image (must end in .exr or .hdr)"
    )
    gamma: float = InputField(
        default=2.2,
        ge=1.0,
        le=3.0,
        description="Gamma value for correction (typically 2.2)"
    )
    brightness: float = InputField(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Brightness multiplier for HDR values"
    )

    def ldr_to_hdr_array(self, ldr_image: Image.Image, gamma: float, brightness: float) -> np.ndarray:
        """Convert LDR image to HDR with proper gamma correction and scaling"""
        # Convert PIL image to float32 numpy array (0-1 range)
        img_array = np.array(ldr_image).astype(np.float32) / 255.0

        # Remove gamma encoding to get linear light values
        # This converts from sRGB-like space to linear space
        linear = np.power(img_array, gamma)

        # Scale to suitable HDR range and apply brightness adjustment
        # This gives us better dynamic range in the HDR output
        hdr = linear * brightness

        # For very bright areas, extend the range more naturally
        # This helps prevent harsh clipping of highlights
        bright_mask = hdr > 1.0
        hdr[bright_mask] = 1.0 + np.log(hdr[bright_mask])

        return hdr

    def ldr_to_exr(self, ldr_image: Image.Image, output_path: str, gamma: float, brightness: float) -> None:
        """Convert LDR image to OpenEXR format and save to disk"""
        # Convert to HDR with proper gamma correction
        hdr_array = self.ldr_to_hdr_array(ldr_image, gamma, brightness)

        # Create header
        header = OpenEXR.Header(ldr_image.size[1], ldr_image.size[0])
        header['compression'] = 3  # ZIP_COMPRESSION
        header['channels'] = {
            'R': {'type': 2},  # FLOAT
            'G': {'type': 2},
            'B': {'type': 2}
        }

        # Prepare channel data
        channels = {
            'R': hdr_array[:, :, 0].tobytes(),
            'G': hdr_array[:, :, 1].tobytes(),
            'B': hdr_array[:, :, 2].tobytes()
        }

        # Write EXR file directly to disk
        exr = OpenEXR.OutputFile(output_path, header)
        exr.writePixels(channels)
        exr.close()

    def ldr_to_hdr(self, ldr_image: Image.Image, output_path: str, gamma: float, brightness: float) -> None:
        """Convert LDR image to Radiance HDR format and save to disk"""
        # Convert to HDR with proper gamma correction
        hdr_array = self.ldr_to_hdr_array(ldr_image, gamma, brightness)

        # Convert to BGR for OpenCV
        bgr_array = cv2.cvtColor(hdr_array, cv2.COLOR_RGB2BGR)

        # Save directly to disk using OpenCV
        cv2.imwrite(output_path, bgr_array)

    def validate_output_path(self, path: str) -> None:
        """Validate the output path is acceptable for HDR files"""
        # Check file extension
        ext = Path(path).suffix.lower()
        if ext not in ['.exr', '.hdr']:
            raise ValueError("Output path must end in .exr or .hdr")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Check write permissions
        if os.path.exists(path):
            if not os.access(path, os.W_OK):
                raise PermissionError(f"No write permission for {path}")
        else:
            try:
                # Try to create/write to the file
                with open(path, 'wb') as f:
                    pass
                os.remove(path)
            except OSError as e:
                raise PermissionError(f"Cannot write to {path}: {e}")

    def invoke(self, context: InvocationContext) -> LDRtoHDROutput:
        """Convert LDR image to HDR and save to disk"""
        # Validate output path first
        self.validate_output_path(self.output_path)

        # Load input image
        logger.info(f"Loading input LDR image: {self.image.image_name}")
        image = context.images.get_pil(self.image.image_name)
        logger.info(f"Loaded PIL image size: {image.size}, mode: {image.mode}")

        try:
            # Determine format from output path
            format = 'exr' if self.output_path.lower().endswith('.exr') else 'hdr'

            # Convert and save based on format
            if format == 'exr':
                logger.info(
                    f"Converting to OpenEXR and saving to: {self.output_path}")
                self.ldr_to_exr(image, self.output_path,
                                self.gamma, self.brightness)
            else:
                logger.info(
                    f"Converting to Radiance HDR and saving to: {self.output_path}")
                self.ldr_to_hdr(image, self.output_path,
                                self.gamma, self.brightness)

            # Get file size for metadata
            file_size = os.path.getsize(self.output_path)
            logger.info("HDR conversion and save completed successfully")

            # Return metadata about the saved file
            return LDRtoHDROutput(
                file_path=self.output_path,
                format=format,
                file_size=file_size
            )

        except Exception as e:
            logger.error(f"HDR conversion failed: {str(e)}")
            raise
