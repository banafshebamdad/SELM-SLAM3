# SELM-SLAM3

SELM-SLAM3 is a deep learning-enhanced RGB-D SLAM framework built upon ORB-SLAM3. It integrates SuperPoint for robust feature extraction and LightGlue for precise feature matching, significantly enhancing localization accuracy in challenging conditions such as low-texture environments and fast-motion scenarios.

Our evaluations on TUM RGB-D, ICL-NUIM, and TartanAir datasets demonstrate SELM-SLAM3's superior tracking accuracy, outperforming ORB-SLAM3 by an average of 87.84% and surpassing state-of-the-art RGB-D SLAM systems by 36.77%. This framework offers a reliable and adaptable solution for real-world assistive navigation applications.

This repository was forked from [SUPERSLAM3](https://github.com/isarlab-department-engineering/SUPERSLAM3), which itself is based on ORB-SLAM3. However, SELM-SLAM3 diverges significantly in its approach to enhancing ORB-SLAM3. While SuperSLAM3 focuses on replacing ORBExtractor with SuperPoint using PyTorch, SELM-SLAM3 implements a new architecture that replaces both feature extraction and matching modules with ONNX-based SuperPoint and LightGlue. Additionally, SELM-SLAM3 eliminates the Bag-of-Words component and implements a new matching strategy for improved performance.

We are currently uploading and setting up the GitHub repository. Stay tuned for updates! *(Last updated: February 6, 2025)*

## Related Paper:
M. Bamdad, H.-P. Hutter, and A. Darvishy. "Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation." 20th International Conference on Computer Vision Theory and Applications (2025).


### Citing:
**If you use SUPERSLAM3 in an academic work, please cite:**
```
M. Bamdad, H.-P. Hutter, and A. Darvishy. "Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation." 20th International Conference on Computer Vision Theory and Applications (2025).
```
Under construction
