# Detection of Micromobility Vehicles in Urban Traffic Videos

![Results Demonstration](data/media/video_results_fgfa_yolo.gif)

This project introduces a novel object detection model, FGFA-YOLOX, specifically designed to enhance the detection of micromobility vehicles (MMVs) such as e-scooters, bikes, and skateboards in urban traffic videos. By integrating the accuracy and speed of single-frame detection with the enriched temporal features of video-based frameworks, FGFA-YOLOX significantly advances the state-of-the-art in urban mobility analytics.

## Key Contributions

- Introduction of the FGFA-YOLOX model, offering a significant improvement on video object detection models.
- Construction of a comprehensive dataset focused on urban micromobility, aiming to propel further research by making it publicly available.

## Model Architecture

Below is the architecture of our FGFA-YOLOX model, illustrating the innovative integration of spatio-temporal features for enhanced object detection in complex urban environments.

![FGFA-YOLOX Architecture](data/media/micro_mobility_architecture.png)

## Dataset

The PolyMMV dataset is crafted for detecting Micromobility Vehicles in urban environments, comprising:

- Video footage capturing diverse urban traffic scenarios.
- Annotations in three formats: YOLO, Pascal VOC, and COCO.
- Pre-trained model checkpoints, allowing users to benchmark or further fine-tune on the PolyMMV dataset.
- A results folder that contains the results for each model trained.

For accessing these resources, visit [our dataset Drive](https://drive.google.com/drive/folders/1oluAUC_AjTcsOit1YU_MN0GfCkk20n1n?usp=sharing).


## Getting Started

For detailed instructions on setting up the project environment and running the FGFA-YOLOX model, please refer to the [installation guide](./installation.md).


## Acknowledgments 
We extend our gratitude to the developers of mmtracking for their comprehensive platform, which played an important role in the implementation of our proposed FGFA-YOLOX architecture. Their work provided the necessary infrastructure that enabled us to seamlessly integrate our model. For their invaluable contribution to our project and the broader research community, we sincerely thank them.

Learn more about their work at [mmtracking GitHub](https://github.com/open-mmlab/mmtracking).

## Contact
For any questions or issues faced during the execution of code, feel free to reach out to me in: khalil.sabri@polymtl.ca . 


## Citation

If you find our work useful in your research, please use the following BibTeX entry:

```bibtex
@article{sabri2024detection,
  title = {Detection of Micromobility Vehicles in Urban Traffic Videos},
  author = {Sabri, Khalil and Djilali, Célia and Bilodeau, Guillaume-Alexandre and Saunier, Nicolas and Bouachir, Wassim},
  journal = {arXiv preprint arXiv:2402.18503},
  year = {2024},
  url = {https://arxiv.org/abs/2402.18503}
}
