# A New Way to Age Estimation for RGBD Images based on a New Face Detection and Extraction Method for Depth Images

This repository implements an age estimation pipeline for **RGB (color)** and **Depth (D)** images. The system combines face extraction, edge detection, entropy calculation, and normalization to predict an individual's age based on both modalities. The methodology is inspired by the paper _"A New Way to Age Estimation for RGB-D Images based on a New Face Detection and Extraction Method for Depth Images"_.
- Link to the paper:
- https://www.mecs-press.org/ijigsp/ijigsp-v10-n11/v10n11-2.html
## Features

- **Face Detection and Extraction**:
  - For depth images: A novel method using standard deviation filtering and ellipse fitting.
  - For color images: Viola-Jones algorithm is used to identify the face region.
- **Entropy and Edge Detection**:
  - Entropy is calculated from the depth image to capture randomness.
  - Sobel edge detection is applied to the color image for feature extraction.
- **Age Normalization**:
  - Combines entropy and edge values to estimate age, normalized between the youngest and oldest samples in the dataset.
![f3](https://github.com/user-attachments/assets/2aef3265-8023-4f11-b470-7fa6dcf8d278)

![fig4](https://github.com/user-attachments/assets/8bdf8fad-1e67-4f42-a864-5aa13e08ed0e)

- ### Please cite below:
Mousavi, Seyed Muhammad Hossein. "A new way to age estimation for rgb-d images, based on a new face detection and extraction method for depth images." International Journal of Image, Graphics and Signal Processing 10.11 (2018): 10.

- DOI:
- [https://www.mecs-press.org/ijigsp/ijigsp-v10-n11/v10n11-2.html](https://doi.org/10.5815/ijigsp.2018.11.02)
