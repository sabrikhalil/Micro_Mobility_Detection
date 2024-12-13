# Micro Mobility Detection Installation and Usage Guide

This guide explains the steps to install and use the Micro Mobility Detection project. It includes installation instructions, running demos, training models, and testing for evaluations.

## Main File to Use

The main file for interacting with the project is: [Main Notebook](https://github.com/sabrikhalil/Micro_Mobility_Detection/blob/main/Main_Notebook.ipynb).

### Overview of the Notebook Sections

1. **Installation**
   - Installs all required libraries to run the code for training or testing.
   - At the end of the section, run the small demo provided to ensure that the installation is successful.

2. **Run Demos Using Pre-Trained Models**
   - To run demos, you will need test videos. These are available in the drive under `videos/videos_test`. Alternatively, you can choose your own video, place it in the appropriate path, and update the code accordingly.
   - Pre-trained model checkpoints are required for running demos. These can be downloaded from the drive under the `Model_checkpoints` folder.
   - A code snippet is provided at the end of this section to run inference on all videos in a folder. Simply place your test videos in a single directory and use this code.

3. **Training**
   - Training can be done for different models, such as `DFF`, `SELSA`, or `FGFA`.
   - For `FGFA`, the proposed architecture is implemented as `fgfa_yolo`.
     - To train from scratch, comment out the `--resume` line in the code.
     - To continue training from a checkpoint, uncomment the `--resume` line. Checkpoints are available in the drive under `Model_checkpoints`.

4. **Testing**
   - This section is used to evaluate the models and reproduce the accuracies and other data reported in the paper.
   - Checkpoints for evaluation are also available in the drive under `Model_checkpoints`.
   - During training and testing, configuration files are used:
     - **Model Configuration**: `configs/vid/fgfa/fgfa_yolo.py`. This file defines the architecture, input image size, and other model parameters.
     - **Dataset Configuration**: `config/_base_/dataset_custom_fgfa.py`. This file specifies the class names and paths for annotations and video folders used for training and testing.

### Notes on Dataset Preparation

- Training and test videos are available in the drive under `videos/videos_train` and `videos/videos_test`.
- A script is provided at the end of the dataset configuration file to convert videos into frames for training and testing. These frame directories are also defined in the dataset configuration file.
- Ensure that the frame sizes match the annotation files.
- Annotations are provided in two formats: `512x256` and `640x640`. The same annotations are used for `FGFA` and `fgfa_yolo` (our architecture).

## Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/sabrikhalil/Micro_Mobility_Detection.git
   cd Micro_Mobility_Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the demo in the notebook to verify the installation.

## Running Demos with Pre-Trained Models

1. Place your test videos in the `videos/videos_test` folder or provide a custom path in the code.
2. Download the required checkpoints from the `Model_checkpoints` folder in the drive and place them in the appropriate directory.
3. Use the provided code in the notebook to run inference on a single video or batch process all videos in a folder.

## Training

1. Modify the `configs/vid/fgfa/fgfa_yolo.py` configuration file as needed:
   - Update input size, architecture, or other parameters.
   - Ensure paths in the dataset configuration (`config/_base_/dataset_custom_fgfa.py`) point to the correct annotations and frame directories.
2. To train from scratch, comment out the `--resume` line in the training script.
3. To resume training from a checkpoint, uncomment the `--resume` line and provide the path to the checkpoint.

## Testing

1. Use the same checkpoints from `Model_checkpoints` to evaluate the model.
2. Modify the dataset configuration file to ensure paths for test frames and annotations are correctly set.
3. Run the evaluation code to reproduce the results from the paper.

## Additional Notes

- Ensure that the frame sizes match the annotations for both training and testing.
- Annotations are available in two formats: `512x256` and `640x640`.
- Use the script at the end of the dataset configuration file to generate frames from videos if needed.

Feel free to explore and modify the provided code to adapt it to your needs!

