# Pure Python Implementation of jpeg2dct

This is a pure Python alternative to the C++ `jpeg2dct` library that extracts DCT coefficients directly from JPEG files.

## Why Pure Python?

The original `jpeg2dct` is implemented in C++ with Python bindings, which:
- Requires compilation with libjpeg/libjpeg-turbo
- Can be difficult to build on some platforms
- Requires a C++ compiler

This pure Python version:
- ✅ No compilation required
- ✅ Easy installation (just pip install dependencies)
- ✅ Cross-platform compatible
- ✅ Easier to modify and understand
- ⚠️ May be slightly slower for batch processing

## Installation

```bash
pip install jpegio numpy
```

The `jpegio` library provides Python access to libjpeg's DCT coefficient reading functionality.

## Usage

### Basic Usage

```python
from jpeg2dct_pure_python import load_dct_coefficients

# Load DCT coefficients from a file
dct_y, dct_cb, dct_cr = load_dct_coefficients('image.jpg', normalized=True)

print(f"Y  shape: {dct_y.shape}")   # (height_blocks, width_blocks, 64)
print(f"Cb shape: {dct_cb.shape}")  # (height_blocks, width_blocks, 64)
print(f"Cr shape: {dct_cr.shape}")  # (height_blocks, width_blocks, 64)
```

### Load from Memory Buffer

```python
from jpeg2dct_pure_python import load_dct_coefficients_from_buffer

with open('image.jpg', 'rb') as f:
    buffer = f.read()

dct_y, dct_cb, dct_cr = load_dct_coefficients_from_buffer(buffer, normalized=True)
```

### Parameters

- **`filename`** or **`buffer`**: Path to JPEG file or bytes buffer
- **`normalized`** (default=True): 
  - `True`: Multiply coefficients by quantization table (denormalized/dequantized)
  - `False`: Return raw quantized coefficients
- **`channels`** (default=3):
  - `3`: Return Y, Cb, Cr components
  - `1`: Return only Y (luminance), Cb and Cr will be empty

## How It Works

JPEG images store data as DCT (Discrete Cosine Transform) coefficients in 8×8 blocks:

1. **JPEG Compression**: Image → DCT → Quantization → Huffman coding
2. **Traditional Decoding**: Huffman → Dequantization → Inverse DCT → Image
3. **This Library**: Huffman → **Extract DCT coefficients directly**

The library uses `jpegio` to:
- Read the JPEG file structure
- Access the DCT coefficient arrays for each color component (Y, Cb, Cr)
- Optionally multiply by quantization tables to get denormalized values
- Reshape from (H, W, 8, 8) blocks to (H, W, 64) for easier processing

## Comparison with Original jpeg2dct

### Similarities
- Same API design
- Same output format
- Support for normalized/unnormalized coefficients
- Support for file and buffer input

### Differences
| Feature | C++ jpeg2dct | Pure Python |
|---------|--------------|-------------|
| Installation | Requires compilation | Just pip install |
| Speed | Faster | Slightly slower |
| Transcoding | Auto-transcodes non-H2V2 formats | Warns but doesn't transcode |
| Dependencies | libjpeg-turbo, C++ compiler | jpegio, numpy |

### Transcoding Note

The C++ version automatically transcodes JPEG images with non-standard chroma subsampling (not 4:2:0 H2V2) to ensure consistent output. The Python version currently does not do this, so results may differ for images with unusual subsampling like 4:1:1, 4:2:2, etc.

## Example Output

```python
from jpeg2dct_pure_python import load_dct_coefficients

dct_y, dct_cb, dct_cr = load_dct_coefficients('photo.jpg')

# Normalized DCT coefficients:
#   Y  component: shape=(205, 205, 64), dtype=int16, range=[-2040, 2040]
#   Cb component: shape=(103, 103, 64), dtype=int16, range=[-1785, 1785]
#   Cr component: shape=(103, 103, 64), dtype=int16, range=[-1734, 1734]
```

Each block contains 64 DCT coefficients (8×8 flattened), representing frequency components from low (DC) to high frequency.

## Testing

Run the example:

```bash
python jpeg2dct_pure_python.py
```

This will:
1. Load a test JPEG file
2. Extract DCT coefficients (normalized and raw)
3. Test buffer loading
4. Compare with C++ implementation if available

## Use Cases

- **Neural Networks**: Feed DCT coefficients directly to CNNs instead of decoded pixels
- **Image Analysis**: Analyze frequency domain features
- **Compression Research**: Study JPEG compression artifacts
- **Fast Processing**: Skip pixel-domain decoding when working with JPEG-native features

## Performance Notes

For maximum performance with large batches:
- Use the original C++ `jpeg2dct` library
- Consider parallel processing with `multiprocessing`
- Cache results if processing the same images multiple times

## References

- Original paper: [Faster Neural Networks Straight from JPEG](https://openreview.net/forum?id=S1ry6Y1vG) (ICLR 2018)
- Original C++ implementation: https://github.com/uber-research/jpeg2dct
- jpegio library: https://github.com/dwgoon/jpegio

## License

This implementation follows the same license as the original jpeg2dct project.
