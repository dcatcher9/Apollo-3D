# Qualified subtitle mask checkpoint

This is the offline H28 candidate qualified on the RTX 5080 on November 22, 2026. The mixed RGB model, native alpha decoder, and learned one-pixel transition use one normalization of the entire extended OCR rectangle. The decision cutoff is 38/255. The independently tested background-conditioned experiment is stored separately.

## Restore and verify

Extract the checkpoint to its own directory. On the development machine, run the following in that directory.
```powershell
& 'E:/ApolloDev/modelopt-py312/Scripts/python.exe' .\run.py verify
& 'E:/ApolloDev/modelopt-py312/Scripts/python.exe' .\run.py mask extended-line.png mask.png
& 'E:/ApolloDev/modelopt-py312/Scripts/python.exe' .\run.py alpha extended-line.png alpha.png
```

Use `Masker.frame(rgb_band, work_boxes, detector_covers)` for a sequence. Rectangles use native band coordinates with exclusive right and bottom edges. An empty list on a fresh OCR observation clears the history. `Masker.alpha(rgb_crop)` returns the isolated continuous alpha before thresholding.

The `model` directory includes the original three trained stages, ONNX graphs, and same-device TensorRT engines. The portable CPU runner uses equivalent PyTorch/ONNX Runtime inference. The SHA-256 manifest checks the contents. Two synthetic images and four temporal decisions test against the original implementation. The `origin` directory preserves the source routines, qualification receipts, and metrics. The executable, active shaders, and model files for the earlier live development build are backed up separately on the development drive.
