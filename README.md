# Fine-Badminton

Fine-Badminton is a fine-grained badminton dataset for temporal action localization (TAL). It provides frame-level temporal annotations for 29 badminton stroke categories and is designed to support research in fine-grained sports video understanding.

## Authors

Tianyu Wang, Junjie Wu, Jingquan Gao, and Shishuo Li

## Overview

Fine-Badminton contains:

- 10 professionally recorded badminton singles matches
- 980,703 frames
- Approximately 10.72 hours of gameplay
- 9,817 annotated action instances
- 29 fine-grained badminton stroke categories

All annotations are provided at frame level with precise start and end timestamps.

## Dataset Structure

```text
Fine-Badminton/
├── README.md
├── LICENSE
├── annotations/
│   ├── video1_dataframe.csv
│   ├── video2_dataframe.csv
│   └── ...
└── videos/
    ├── video1.mp4
    └── video2.mp4
```

## Data Source

The video data were collected from publicly available badminton broadcasts on YouTube, including international tournaments and amateur competitions. Copyright of the original videos remains with the respective content owners.

Only processed video clips and frame-level temporal annotations are distributed.

## Action Categories

1. Short Serve
2. Overhead Cross-court Net Shot
3. Lift
4. Drive
5. Drop Shot
6. Flat Clear
7. Cross-court Net Return of Serve
8. Smash(Full-length)
9. Soft Net Shot Return of Serve
10. Net Shot
11. Jump Smash
12. Lift Return of Smash
13. Lift Return of Serve
14. Defensive Lift
15. Underhead Cross-court Net Shot
16. Drop Shot Return of Smash
17. Net Kill
18. High Clear
19. Soft Net Shot
20. Flat Serve
21. Clear Return of Serve
22. Short Smash
23. Cross-court Net Return of Smash
24. Net Shot Return of Serve
25. High Serve
26. Drive Return of Smash
27. Smash Return of Serve
28. Push Shot
29. Drop Shot Return of Serve

## Annotation Principle

Each action is annotated according to the complete stroke process:

- **Start time:** the first frame in which the player initiates the stroke motion.
- **End time:** the frame in which the racket–shuttle interaction and immediate follow-through are completed.

## Naming Convention

The category names were defined through consultations with professional badminton athletes to ensure consistency with standard badminton terminology.

## Intended Use

Fine-Badminton is intended for research in:

- Temporal Action Localization (TAL)
- Fine-Grained Action Recognition
- Sports Video Understanding
- Temporal Segmentation
- Video Representation Learning

## Data Usage Agreement

### Purpose

The Fine-Badminton dataset is released to support academic research in temporal action localization, sports video understanding, and related fields.

### Permitted Uses

Users may:

- Download and use the dataset for non-commercial research and educational purposes;
- Train and evaluate machine learning models;
- Publish research results derived from the dataset.

### Restrictions

Users may not:

- Redistribute the dataset or any substantial portion without prior written permission;
- Use the dataset for commercial purposes;
- Attempt to identify or contact individuals appearing in the videos;
- Misrepresent the source or ownership of the dataset.

### Copyright Notice

The video data were collected from publicly available badminton broadcasts on YouTube. Copyright of the original videos remains with the respective content owners.

Fine-Badminton is distributed solely for non-commercial academic research purposes and includes processed video clips and frame-level temporal annotations.

### Disclaimer

The dataset is provided "as is" without warranty of any kind. The authors are not liable for any damages arising from its use.

### Acceptance

By downloading or using the dataset, users agree to the terms of this agreement.

## License

This dataset is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0).

For the full license text, see the `LICENSE` file or visit:

https://creativecommons.org/licenses/by-nc/4.0/

## Citation

If you use Fine-Badminton in your research, please cite:

```
@article{wang2026fine-badminton,
  title={Fine-Badminton: A Fine-Grained Badminton Dataset for Temporal Action Localization},
  author={Wang, Tianyu and Wu, Junjie and Gao, Jingquan and Li, Shishuo},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  note={Under review}
}
```

## Contact

For questions regarding dataset usage, please contact: sy2408214@buaa.edu.cn