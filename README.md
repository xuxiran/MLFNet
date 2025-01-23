Here is the revised README with a clear hierarchical structure:

---

# All Codes

This repository contains all the codes implemented for this study.

### Folder Structure

- **dataset**: Contains the datasets used in this study. Please download the KUL, DTU, and PKU datasets.
  - **KUL**: [https://zenodo.org/record/3377911](https://zenodo.org/record/3377911)
  - **DTU**: [https://zenodo.org/record/3618205](https://zenodo.org/record/3618205)
  - **PKU**: You can contact Prof. Chen for non-commercial use of this dataset (Fu et al., 2021).

  For more information, please refer to the `readme.md` file within this folder.

- **preprocess_code**: Contains preprocessing codes for the three datasets used in this study.
  - `preprocess_DTU.m`
  - `preprocess_KUL.m`
  - `preprocess_PKU.m`

- **preprocess_data**: Contains the preprocessed data obtained after running the above `.m` files.

- **MLFNet**: Contains the complete code for MLFNet. For more information, please refer to the `readme.md` file within this folder.

---


Fu, Z., Wang, B., Wu, X., & Chen, J. (2021). Auditory Attention Decoding from EEG using Convolutional Recurrent Neural Network. 29th European Signal Processing Conference (EUSIPCO), 970–974. https://doi.org/10.23919/EUSIPCO54536.2021.9616195
