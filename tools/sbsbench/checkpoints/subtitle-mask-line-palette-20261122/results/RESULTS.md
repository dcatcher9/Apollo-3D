# Direct Glyph subtitle mask comparison

The candidate keeps the mixed-precision H28 encoder. A learned head combines the source pixel, colors recurring across the line and two estimates of the surrounding background. It returns continuous fill-plus-outline protection at native resolution. Every language uses the same 38/255 mask threshold. The input is the entire working rectangle, expanded on each side by one sixteenth of its height and clipped to the original OCR cover.

## New captions

On six new English, Chinese and Japanese caption sequences, aggregate fill-plus-outline F1 rose from 87.33% to 94.22%. Recall rose from 79.22% to 91.08%, and precision rose from 97.28% to 97.57%. The 42 frames use native libass labels and new camera-frame windows. The OCR rectangles are calibrated proxies. Some source recordings also appear in earlier tests at different timestamps. [Paired scores](paired.json) · [Fixture provenance](manifest.json)

| Fresh test | Saved H28 | Candidate |
| --- | ---: | ---: |
| English F1 | 88.45% | 92.55% |
| Chinese F1 | 84.45% | 87.75% |
| Japanese F1 | 87.26% | 95.87% |
| Same-cue target pixels that flipped | 774 / 323,975 | 548 / 323,975 |
| Same-cue clear pixels that flipped | 1,237 | 728 |
| Old-only pixels selected on same-box swaps | 323 / 23,333 | 420 / 23,333 |
| Old-only pixels selected after disappearance | 0 / 104,161 | 0 / 104,161 |
| Clear background pixels selected | 11,920 | 12,186 |

All six swaps rejected history. The additional selected pixels on those frames came from the current-frame prediction. The English example with similar fill and outline colors selected 5,943 clear pixels, up from 173. The dark English example reduced that error from 7,893 to 463. The close crops below show the onset frames. [Frame-level decisions](palette-sixteenth.json)

![Fresh subtitle crops](fresh-comparison.png)

## Real captures and actual OCR

On 24 archived captures with marked glyph pixels, 75 of the 236 marked outline pixels lay outside the previous working crop. Every one was inside the original detector cover. On a separate failing Chinese camera onset, all labeled glyph pixels were inside the working crop. Rerendering with only the outline color changed shifted both fill recall and background errors, establishing a separate appearance failure. [Capture analysis](../rawmatte-line-palette-1122/marked-diagnosis.json) · [Controlled color changes](../rawmatte-spatial-guide-1122/controlled-contrast.json)

| Native marked pixels | Earlier live | Saved H28 | Candidate |
| --- | ---: | ---: | ---: |
| Fill selected / 299 | 47 | 297 | 299 |
| Thin fill selected / 63 | 28 | 63 | 63 |
| Outline selected / 236 | 115 | 108 | 150 |
| Gaps incorrectly selected / 119 | 22 | 10 | 7 |
| Background incorrectly selected / 233 | 0 | 0 | 3 |

On the 30-frame 4K sequence with recorded OCR, F1 was 88.70% for the earlier live model, 94.96% for saved H28 and 96.09% for the candidate. Target flicker was 0.846%, 0.327% and 0.378%, respectively. Expanding the work by one quarter of its height recovered more marked outlines but raised total 4K background errors to 219,494. The selected one-sixteenth margin produced 103,517. [Marked captures](../rawmatte-detector-margin-fine-1122/captured-verified.json) · [4K sequence](../rawmatte-detector-margin-fine-1122/sequence-verified.json)

## GPU verification

The offline D3D11/TensorRT runner processed the native FP16 source, normalized it, ran the mixed CNN, reconstructed full lines from tiles, and evaluated the learned native decoder on the RTX 5080. All 339,000 final binary pixel decisions agreed with the reference across seven inputs. This measures single-frame mask correctness. Streaming queue behavior, causal history and headset rendering require host-path verification. [GPU parity](../rawmatte-palette-gpu-1122/parity.json)
