# Restore the line-palette subtitle mask

This is the November 22 mixed H28 candidate with the learned native-resolution palette decoder. It uses a protection threshold of 38/255 in English, Chinese and Japanese. Each OCR work rectangle grows on all four sides by one sixteenth of its height, rounded up and clipped to the detector cover. The encoder normalizes that entire rectangle. The temporal tracker checks the current cue before using its previous mask.

## Restore a working copy

Run these commands in PowerShell. Choose a fresh destination.

```powershell
$Candidate = 'E:\ApolloDev\restored-line-palette-h28'
Expand-Archive -LiteralPath 'E:\ApolloDev\checkpoints\subtitle-mask-line-palette-20261122\line-palette-h28.zip' -DestinationPath $Candidate -ErrorAction Stop
& 'E:\ApolloDev\modelopt-py312\Scripts\python.exe' -B "$Candidate\run.py" verify
```

Verification checks each archived file, five native opacity maps and the masks from a seven-frame subtitle sequence including onset, replacement and disappearance. Expected outputs come from the original experiment. To inspect a new native RGB subtitle crop, use `run.py alpha input.png alpha.png` or `run.py mask input.png mask.png`. The `Masker.frame` API takes the native RGB band and the current OCR working rectangles and detector covers in left, top, right, bottom order. Supply empty lists for a blank observation.

## Recover from Git

The local tag is `subtitle-mask/line-palette-h28-20261122`. The checkpoint also resides in that tag under `tools/sbsbench/checkpoints/subtitle-mask-line-palette-20261122`. To extract from Git into a fresh folder without switching the active branch, use the following commands.

```powershell
$Recovered = 'E:\ApolloDev\recovered-line-palette-git'
$Packed = 'E:\ApolloDev\recovered-line-palette-git.zip'
$Tree = 'tools/sbsbench/checkpoints/subtitle-mask-line-palette-20261122'
git -C 'E:\Git\Repo\Apollo-3D' archive --format=zip -o $Packed "subtitle-mask/line-palette-h28-20261122:$Tree"
if ($LASTEXITCODE -ne 0) { throw 'Git archive failed' }
Expand-Archive -LiteralPath $Packed -DestinationPath $Recovered -ErrorAction Stop
& 'E:\ApolloDev\modelopt-py312\Scripts\python.exe' -B "$Recovered\run.py" verify
```

The archive includes the runtime ONNX encoder, the RTX 5080 TensorRT engine, the native decoder weights, the production shader sources, normalization, tiling and temporal logic. The standalone runner uses ONNX Runtime and PyTorch. The independent GPU comparisons are recorded in `results/native-96.json`. Live host deployment is a separate step. The preceding baseline and its live host archive remain under `E:\ApolloDev\checkpoints\subtitle-mask-20261122`.
