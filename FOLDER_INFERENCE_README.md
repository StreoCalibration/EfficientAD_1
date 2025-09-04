# EfficientAD Folder-Based Inference

This document describes the folder-based inference functionality that allows you to process entire directories of images and get pixel-level anomaly scores normalized to 0-1 range.

## 🚀 Quick Start

### Basic Usage

```bash
# Process a batch folder from the train data
python example_folder_inference.py --model_path "path/to/model.ckpt" --batch_folder "batch_01"

# Process any custom folder
python example_folder_inference.py --model_path "path/to/model.ckpt" --custom_folder "/path/to/images"
```

### Advanced Usage

```bash
# Full control over processing
python tools/folder_inference.py \
  --model_path "path/to/model.ckpt" \
  --folder_path "/path/to/images" \
  --output_dir "my_results" \
  --image_size 256 256 \
  --batch_size 16 \
  --save_as_image True \
  --save_as_npy True
```

## 📁 File Structure

```
EfficientAD_1/
├── tools/
│   ├── folder_inference.py      # Core folder processing script
│   └── inference.py             # Original single-image inference
├── example_folder_inference.py  # Easy-to-use example script
├── test_folder_inference.py     # Test script
└── FOLDER_INFERENCE_README.md   # This file
```

## 🔧 Features

### 1. **Folder Processing**
- Processes entire directories of images automatically
- Supports common image formats: PNG, JPG, JPEG, BMP, TIFF
- Maintains consistent ordering (sorted by filename)
- Batch processing for memory efficiency

### 2. **Normalized Output (0-1 Range)**
- All pixel values in anomaly maps are normalized to 0-1 range
- 0 = Normal (no anomaly)
- 1 = Maximum anomaly score
- Consistent scaling across all images

### 3. **Multiple Output Formats**
- **PNG files**: Color-mapped visualization images
- **NPY files**: Raw normalized numpy arrays for exact values
- **Summary file**: Detailed statistics and results

### 4. **Batch Folder Integration**
- Direct support for the batch folders created by the data splitting
- Easy processing of `batch_01`, `batch_02`, etc.

## 📊 Output Structure

When you run folder inference, you'll get:

```
inference_results_batch_01/
├── 0_anomaly.png           # Visualization image
├── 0_anomaly.npy           # Raw normalized values (0-1)
├── 1_anomaly.png
├── 1_anomaly.npy
├── ...
└── results_summary.txt     # Detailed statistics
```

### Results Summary File
Contains comprehensive information:
- Processing parameters
- Per-image statistics (anomaly scores, pixel statistics)
- Overall dataset statistics

## 🔍 Understanding the Output

### Anomaly Scores
Each image gets multiple scores:
1. **Overall Anomaly Score**: Maximum value in the anomaly map
2. **Max Pixel Score**: Highest anomaly value (0-1) in the image
3. **Mean Pixel Score**: Average anomaly value across all pixels

### Pixel-Level Anomaly Maps
- Each pixel has a value between 0 and 1
- Higher values indicate higher anomaly likelihood
- Perfect for:
  - Precise anomaly localization
  - Creating custom thresholds
  - Further analysis and processing

## 📋 Usage Examples

### Example 1: Process First Batch
```python
# Process the first 10,000 images
python example_folder_inference.py \
  --model_path "checkpoints/best_model.ckpt" \
  --batch_folder "batch_01"
```

### Example 2: Custom Processing
```python
# Process with custom settings
python tools/folder_inference.py \
  --model_path "checkpoints/best_model.ckpt" \
  --folder_path "Z:/Data/1_Work/Zeta/20250903_Training/train/batch_01" \
  --output_dir "detailed_results_batch01" \
  --image_size 512 512 \
  --batch_size 4 \
  --save_as_image True \
  --save_as_npy True
```

### Example 3: Memory-Optimized Processing
```python
# For limited GPU memory
python tools/folder_inference.py \
  --model_path "checkpoints/best_model.ckpt" \
  --folder_path "large_dataset/" \
  --batch_size 2 \
  --image_size 256 256
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model_path` | Path to trained model checkpoint | Required |
| `--folder_path` | Input folder containing images | Required |
| `--output_dir` | Output directory for results | Auto-generated |
| `--image_size` | Processing image size (H W) | [256, 256] |
| `--batch_size` | Processing batch size | 8 |
| `--save_as_image` | Save PNG visualizations | True |
| `--save_as_npy` | Save numpy arrays | True |

## 🧪 Testing

Run the test script to verify functionality:

```bash
python test_folder_inference.py
```

This will:
- Test normalization functions
- Create sample anomaly maps
- Verify batch folder access
- Generate test outputs in `test_anomaly_outputs/`

## 💡 Tips and Best Practices

### Memory Management
- Reduce `batch_size` if you get GPU out-of-memory errors
- Monitor GPU memory usage during processing
- Process large datasets in smaller chunks if needed

### File Organization
- Use descriptive output directory names
- Keep model checkpoints organized
- Archive processed results for later analysis

### Performance Optimization
- Use appropriate batch sizes for your hardware
- Consider image size vs. accuracy trade-offs
- Use SSD storage for faster I/O when possible

## 🔬 Technical Details

### Normalization Process
1. Compute raw anomaly map from model
2. Find min/max values in the map
3. Apply linear scaling: `(value - min) / (max - min)`
4. Handle edge case where all values are identical (set to 0)

### Supported Image Formats
- PNG (recommended for training data)
- JPEG/JPG
- BMP
- TIFF

### GPU/CPU Support
- Automatically detects and uses available GPU
- Falls back to CPU if CUDA not available
- Memory-efficient batch processing

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce batch size
   --batch_size 2
   ```

2. **No Images Found**
   - Check folder path is correct
   - Verify image file extensions
   - Ensure read permissions

3. **Model Loading Error**
   - Verify model checkpoint path
   - Check model compatibility
   - Ensure all dependencies installed

### Getting Help

1. Run test script: `python test_folder_inference.py`
2. Check output directory permissions
3. Verify model checkpoint integrity
4. Monitor system resource usage

## 📈 Performance Expectations

### Processing Speed (approximate)
- **CPU**: ~1-5 images/second
- **GPU (RTX 3080)**: ~20-50 images/second
- **Batch processing**: Significantly faster than individual images

### Memory Usage
- **GPU**: ~2-8GB depending on batch size and image size
- **RAM**: ~1-4GB for loading and processing

### Disk Space
- **PNG outputs**: ~100KB - 1MB per image
- **NPY outputs**: ~256KB per 256x256 image
- **Summary files**: ~1-10KB depending on dataset size

## 🎯 Next Steps

After running folder inference:

1. **Analyze Results**: Review `results_summary.txt` for insights
2. **Set Thresholds**: Use pixel scores to define anomaly thresholds  
3. **Visualize Patterns**: Examine PNG outputs for anomaly patterns
4. **Further Processing**: Use NPY files for custom analysis
5. **Performance Tuning**: Adjust parameters based on results

---

This folder-based inference system makes it easy to process large datasets and get consistent, normalized anomaly scores for every pixel in your images!