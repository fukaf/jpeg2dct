"""
Pure Python implementation to extract DCT coefficients from JPEG files.

This script provides an alternative to the C++ jpeg2dct library using pure Python.
It uses the 'jpegio' library which wraps libjpeg to access DCT coefficients directly.

Installation:
    pip install jpegio

Usage:
    from jpeg2dct_pure_python import load_dct_coefficients
    
    dct_y, dct_cb, dct_cr = load_dct_coefficients('image.jpg', normalized=True)
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import jpegio as jio
except ImportError:
    raise ImportError(
        "jpegio is required for this script. Install it with: pip install jpegio"
    )


def load_dct_coefficients(
    filename: str,
    normalized: bool = True,
    channels: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DCT coefficients from a JPEG file.
    
    Args:
        filename: Path to the JPEG file
        normalized: If True, multiply DCT coefficients by quantization table values
                   (denormalized). If False, return raw quantized coefficients.
        channels: Number of channels to read (1 for grayscale, 3 for color)
    
    Returns:
        Tuple of three numpy arrays (Y, Cb, Cr) containing DCT coefficients.
        Each array has shape (height_in_blocks, width_in_blocks, 64)
        For grayscale images or when channels=1, Cb and Cr will be empty arrays.
    """
    # Read JPEG structure
    jpeg = jio.read(filename)
    
    # Get DCT coefficients and quantization tables
    coef_arrays = jpeg.coef_arrays
    quant_tables = jpeg.quant_tables
    
    # Number of components in the image
    num_components = len(coef_arrays)
    
    # Check if image needs transcoding (non-standard chroma subsampling)
    if num_components > 1:
        comp_info = jpeg.comp_info
        is_h2v2 = (
            comp_info[0]['v_samp_factor'] == 2 and
            comp_info[0]['h_samp_factor'] == 2 and
            comp_info[1]['v_samp_factor'] == 1 and
            comp_info[1]['h_samp_factor'] == 1 and
            comp_info[2]['v_samp_factor'] == 1 and
            comp_info[2]['h_samp_factor'] == 1
        )
        
        if not is_h2v2:
            warnings.warn(
                "Non-standard JPEG chroma subsampling detected. "
                "Results may differ from jpeg2dct C++ implementation which transcodes such images.",
                UserWarning
            )
    
    def process_component(comp_idx: int) -> np.ndarray:
        """Process a single color component."""
        if comp_idx >= num_components:
            # Create empty component (happens when requesting 3 channels from grayscale)
            if num_components > 0:
                # Make it half size of the first component (similar to C++ version)
                h, w = coef_arrays[0].shape[:2]
                h_half = (h + 1) // 2
                w_half = (w + 1) // 2
                return np.zeros((h_half, w_half, 64), dtype=np.int16)
            else:
                return np.zeros((0, 0, 64), dtype=np.int16)
        
        # Get DCT coefficients for this component
        dct_blocks = coef_arrays[comp_idx]  # Shape: (height_in_blocks, width_in_blocks, 8, 8)
        
        # Get quantization table index for this component
        quant_idx = jpeg.comp_info[comp_idx]['quant_tbl_no']
        quant_table = quant_tables[quant_idx]  # Shape: (8, 8)
        
        # Reshape from (H, W, 8, 8) to (H, W, 64)
        h_blocks, w_blocks = dct_blocks.shape[:2]
        dct_flat = dct_blocks.reshape(h_blocks, w_blocks, 64)
        
        if normalized:
            # Multiply by quantization table values (denormalization)
            # This reverses the quantization: original_dct = quantized_dct * quant_table
            quant_flat = quant_table.reshape(64)
            dct_denorm = dct_flat.astype(np.int32) * quant_flat.astype(np.int32)
            return dct_denorm.astype(np.int16)
        else:
            # Return raw quantized coefficients
            return dct_flat.astype(np.int16)
    
    # Process each component
    if channels == 3:
        dct_y = process_component(0)
        dct_cb = process_component(1)
        dct_cr = process_component(2)
    else:
        dct_y = process_component(0)
        dct_cb = np.zeros((0, 0, 64), dtype=np.int16)
        dct_cr = np.zeros((0, 0, 64), dtype=np.int16)
    
    return dct_y, dct_cb, dct_cr


def load_dct_coefficients_from_buffer(
    buffer: bytes,
    normalized: bool = True,
    channels: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load DCT coefficients from a JPEG buffer in memory.
    
    Args:
        buffer: Bytes object containing JPEG data
        normalized: If True, multiply DCT coefficients by quantization table values
        channels: Number of channels to read (1 for grayscale, 3 for color)
    
    Returns:
        Tuple of three numpy arrays (Y, Cb, Cr) containing DCT coefficients.
    """
    import io
    import tempfile
    
    # jpegio doesn't support reading from buffers directly, so we use a temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(buffer)
        tmp_path = tmp.name
    
    try:
        result = load_dct_coefficients(tmp_path, normalized=normalized, channels=channels)
    finally:
        import os
        os.unlink(tmp_path)
    
    return result


def main():
    """Example usage and comparison with jpeg2dct C++ implementation."""
    import os
    
    # Example file
    test_file = os.path.join(os.path.dirname(__file__), 'test', 'data', 'DCT_16_16.jpg')
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please provide a JPEG file path.")
        return
    
    print("Loading DCT coefficients from:", test_file)
    print()
    
    # Load with normalization
    dct_y, dct_cb, dct_cr = load_dct_coefficients(test_file, normalized=True)
    
    print("Normalized DCT coefficients:")
    print(f"  Y  component: shape={dct_y.shape}, dtype={dct_y.dtype}, range=[{dct_y.min()}, {dct_y.max()}]")
    print(f"  Cb component: shape={dct_cb.shape}, dtype={dct_cb.dtype}, range=[{dct_cb.min()}, {dct_cb.max()}]")
    print(f"  Cr component: shape={dct_cr.shape}, dtype={dct_cr.dtype}, range=[{dct_cr.min()}, {dct_cr.max()}]")
    print()
    
    # Load without normalization
    dct_y_raw, dct_cb_raw, dct_cr_raw = load_dct_coefficients(test_file, normalized=False)
    
    print("Raw (quantized) DCT coefficients:")
    print(f"  Y  component: shape={dct_y_raw.shape}, dtype={dct_y_raw.dtype}, range=[{dct_y_raw.min()}, {dct_y_raw.max()}]")
    print(f"  Cb component: shape={dct_cb_raw.shape}, dtype={dct_cb_raw.dtype}, range=[{dct_cb_raw.min()}, {dct_cb_raw.max()}]")
    print(f"  Cr component: shape={dct_cr_raw.shape}, dtype={dct_cr_raw.dtype}, range=[{dct_cr_raw.min()}, {dct_cr_raw.max()}]")
    print()
    
    # Test buffer loading
    print("Testing buffer loading...")
    with open(test_file, 'rb') as f:
        buffer = f.read()
    dct_y_buf, dct_cb_buf, dct_cr_buf = load_dct_coefficients_from_buffer(buffer, normalized=True)
    print(f"  Buffer loading successful: Y shape={dct_y_buf.shape}")
    print()
    
    # Try to compare with original jpeg2dct if available
    try:
        from jpeg2dct.numpy import load as jpeg2dct_load
        
        print("Comparing with jpeg2dct C++ implementation...")
        dct_y_cpp, dct_cb_cpp, dct_cr_cpp = jpeg2dct_load(test_file, normalized=True)
        
        print(f"  C++ Y  shape: {dct_y_cpp.shape}, Python Y  shape: {dct_y.shape}")
        print(f"  C++ Cb shape: {dct_cb_cpp.shape}, Python Cb shape: {dct_cb.shape}")
        print(f"  C++ Cr shape: {dct_cr_cpp.shape}, Python Cr shape: {dct_cr.shape}")
        
        if np.array_equal(dct_y, dct_y_cpp):
            print("  ✓ Y coefficients match perfectly!")
        else:
            diff = np.abs(dct_y.astype(np.int32) - dct_y_cpp.astype(np.int32))
            print(f"  ✗ Y coefficients differ: max diff={diff.max()}, mean diff={diff.mean():.2f}")
        
    except ImportError:
        print("jpeg2dct C++ implementation not available for comparison.")


if __name__ == '__main__':
    main()
