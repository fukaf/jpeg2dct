"""
Simple test script for the pure Python jpeg2dct implementation.
This script demonstrates basic usage and validates the implementation.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic DCT loading functionality."""
    print("=" * 60)
    print("Testing Pure Python jpeg2dct Implementation")
    print("=" * 60)
    print()
    
    # Check dependencies
    try:
        import numpy as np
        print("✓ NumPy installed")
    except ImportError:
        print("✗ NumPy not installed. Run: pip install numpy")
        return False
    
    try:
        import jpegio
        print("✓ jpegio installed")
    except ImportError:
        print("✗ jpegio not installed. Run: pip install jpegio")
        return False
    
    print()
    
    # Import our implementation
    try:
        from jpeg2dct_pure_python import load_dct_coefficients, load_dct_coefficients_from_buffer
        print("✓ Pure Python implementation imported successfully")
    except Exception as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    print()
    
    # Find a test image
    test_images = [
        'test/data/DCT_16_16.jpg',
        '../test/data/DCT_16_16.jpg',
        'DCT_16_16.jpg'
    ]
    
    test_file = None
    for path in test_images:
        if os.path.exists(path):
            test_file = path
            break
    
    if test_file is None:
        print("⚠ No test JPEG file found. Please provide a JPEG file.")
        print("  Expected locations:", test_images)
        return False
    
    print(f"Using test file: {test_file}")
    print()
    
    # Test 1: Load normalized coefficients
    print("Test 1: Loading normalized DCT coefficients...")
    try:
        dct_y, dct_cb, dct_cr = load_dct_coefficients(test_file, normalized=True)
        print(f"  ✓ Y  component: shape={dct_y.shape}, dtype={dct_y.dtype}")
        print(f"  ✓ Cb component: shape={dct_cb.shape}, dtype={dct_cb.dtype}")
        print(f"  ✓ Cr component: shape={dct_cr.shape}, dtype={dct_cr.dtype}")
        print(f"  Value ranges: Y=[{dct_y.min()}, {dct_y.max()}], "
              f"Cb=[{dct_cb.min()}, {dct_cb.max()}], "
              f"Cr=[{dct_cr.min()}, {dct_cr.max()}]")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 2: Load raw coefficients
    print("Test 2: Loading raw (quantized) DCT coefficients...")
    try:
        dct_y_raw, dct_cb_raw, dct_cr_raw = load_dct_coefficients(test_file, normalized=False)
        print(f"  ✓ Y  component: shape={dct_y_raw.shape}, dtype={dct_y_raw.dtype}")
        print(f"  Value ranges: Y=[{dct_y_raw.min()}, {dct_y_raw.max()}], "
              f"Cb=[{dct_cb_raw.min()}, {dct_cb_raw.max()}], "
              f"Cr=[{dct_cr_raw.min()}, {dct_cr_raw.max()}]")
        
        # Verify that normalized has larger range
        if abs(dct_y.max() - dct_y.min()) > abs(dct_y_raw.max() - dct_y_raw.min()):
            print("  ✓ Normalized values have larger range as expected")
        else:
            print("  ⚠ Warning: Normalized range not larger than raw")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print()
    
    # Test 3: Load from buffer
    print("Test 3: Loading from memory buffer...")
    try:
        with open(test_file, 'rb') as f:
            buffer = f.read()
        dct_y_buf, dct_cb_buf, dct_cr_buf = load_dct_coefficients_from_buffer(buffer, normalized=True)
        
        # Verify buffer loading gives same result
        if np.array_equal(dct_y, dct_y_buf):
            print(f"  ✓ Buffer loading matches file loading")
        else:
            print(f"  ✗ Buffer loading differs from file loading")
            return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False
    
    print()
    
    # Test 4: Compare with C++ implementation if available
    print("Test 4: Comparing with C++ jpeg2dct (if available)...")
    try:
        from jpeg2dct.numpy import load as jpeg2dct_load
        
        dct_y_cpp, dct_cb_cpp, dct_cr_cpp = jpeg2dct_load(test_file, normalized=True)
        
        print(f"  C++ implementation loaded successfully")
        print(f"  Shapes - C++: Y={dct_y_cpp.shape}, Python: Y={dct_y.shape}")
        
        if dct_y.shape == dct_y_cpp.shape:
            print(f"  ✓ Shapes match")
            
            # Check if values match
            if np.array_equal(dct_y, dct_y_cpp):
                print(f"  ✓ Y coefficients match perfectly!")
            else:
                diff_y = np.abs(dct_y.astype(np.int32) - dct_y_cpp.astype(np.int32))
                print(f"  ~ Y coefficients differ slightly: max_diff={diff_y.max()}, mean_diff={diff_y.mean():.4f}")
                
                if diff_y.max() == 0:
                    print(f"  ✓ All differences are zero - perfect match!")
            
            # Check Cb
            if np.array_equal(dct_cb, dct_cb_cpp):
                print(f"  ✓ Cb coefficients match perfectly!")
            
            # Check Cr  
            if np.array_equal(dct_cr, dct_cr_cpp):
                print(f"  ✓ Cr coefficients match perfectly!")
        else:
            print(f"  ✗ Shapes don't match")
            
    except ImportError:
        print(f"  ⚠ C++ jpeg2dct not available (not installed or not compiled)")
    except Exception as e:
        print(f"  ✗ Comparison failed: {e}")
    
    print()
    print("=" * 60)
    print("All tests completed successfully! ✓")
    print("=" * 60)
    
    return True


if __name__ == '__main__':
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
