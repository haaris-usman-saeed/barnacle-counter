| Variant | AP img2 | AP img1 | Mean AP | Calibrated F1 img2 | Calibrated F1 img1 | Count img2 (397) | Count img1 (1,731) |
|---|---|---|---|---|---|---|---|
| C0 current: ring+LoG, base features, HGB | 0.310 | 0.553 | 0.432 | 0.335 ± 0.055 | 0.547 ± 0.014 | 268 ± 106 | 1516 ± 402 |
| C1 + per-photo feature standardization | 0.231 | 0.526 | 0.379 | 0.283 ± 0.028 | 0.519 ± 0.031 | 395 ± 241 | 1725 ± 730 |
| C2 + wide HOG and radial gradients | 0.331 | 0.574 | 0.453 | 0.387 ± 0.019 | 0.574 ± 0.015 | 507 ± 249 | 1852 ± 490 |
| C3 C2 + per-photo standardization | 0.277 | 0.566 | 0.422 | 0.346 ± 0.026 | 0.559 ± 0.021 | 444 ± 186 | 1441 ± 227 |
| C4 C1 features, random forest | 0.254 | 0.532 | 0.393 | 0.305 ± 0.040 | 0.540 ± 0.022 | 532 ± 326 | 1546 ± 361 |
| C5 C1 + contrast-jitter augmentation | 0.251 | 0.626 | 0.439 | 0.309 ± 0.019 | 0.593 ± 0.026 | 328 ± 152 | 1730 ± 624 |
| C6 C1 on union (ring+LoG + Hough) candidates | 0.295 | 0.638 | 0.466 | 0.316 ± 0.062 | 0.608 ± 0.020 | 431 ± 369 | 1473 ± 278 |
| C7 C2 features on Hough candidates | 0.404 | 0.696 | 0.550 | 0.358 ± 0.096 | 0.670 ± 0.034 | 495 ± 477 | 1907 ± 584 |
| C8 C2 features on union candidates | 0.388 | 0.656 | 0.522 | 0.397 ± 0.042 | 0.628 ± 0.031 | 288 ± 110 | 2024 ± 678 |
| C9 C2 + contrast-jitter augmentation | 0.380 | 0.611 | 0.496 | 0.420 ± 0.019 | 0.596 ± 0.018 | 545 ± 260 | 1540 ± 382 |
