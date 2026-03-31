# Intent and Trajectory Prediction

## Project Overview
In an L4 urban environment, reacting to where a pedestrian or cyclist is currently located is not enough; autonomous systems must accurately predict where they will be in the future. 

This project tackles **Problem Statement 1: Intent & Trajectory Prediction**. It implements a deep learning model designed to predict the future coordinates of dynamic agents (pedestrians and cyclists) over the next **3 seconds** based on their past **2 seconds** of motion. The system generates multi-modal predictions, outputting the 3 most likely future paths while accounting for the social context of the scene.

## Model Architecture
The solution uses a custom PyTorch architecture, `IntentAndTrajectoryPredictor`, which consists of the following key components:

* **Temporal Encoder (LSTM):** A Long Short-Term Memory (LSTM) network processes the past 2 seconds of spatial coordinates (at 2Hz) to capture the agent's movement history and momentum.
* **Social Attention Mechanism:** A custom attention layer that weights the importance of surrounding pedestrians and cyclists. This allows the model to understand "Social Context" and how agents avoid each other in shared spaces.
* **Feature Fusion:** Combines the ego-agent's intent vector with the social context vector into a unified dense representation.
* **Multi-Modal Prediction Heads:** * **Regression Head:** Predicts 3 distinct future trajectories (6 future time steps each).
    * **Classification Head:** Outputs a confidence score/probability for each of the 3 predicted modes.

## Dataset Used
This project utilizes the **[nuScenes Dataset](https://www.nuscenes.org/nuscenes)**, a large-scale autonomous driving dataset. 

* **Features Extracted:** Past trajectories (2s), future ground truth trajectories (3s), and kinematic data (velocity, acceleration, yaw rate).
* **Split Used:** The code is configured to run on the `v1.0-mini` split for rapid prototyping, but can be scaled to the full `v1.0-trainval` split.

## Setup & Installation Instructions

### Prerequisites
* Python 3.8+
* Google Colab (recommended for GPU acceleration) or a local machine with a CUDA-enabled GPU.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Intent_and_trajectory.git](https://github.com/YourUsername/Intent_and_trajectory.git)
    cd Intent_and_trajectory
    ```
2.  **Install the required dependencies:**
    The primary dependencies are PyTorch and the official nuScenes devkit.
    ```bash
    pip install torch torchvision
    pip install nuscenes-devkit
    pip install matplotlib numpy
    ```
3.  **Data Setup:**
    Download the nuScenes v1.0-mini dataset and extract it. Update the `DATAROOT` variable in the notebook to point to your local dataset path or Google Drive mount.

## How to Run the Code
The entire training and evaluation pipeline is contained within the `Intent_and_trajectory.ipynb` Jupyter Notebook.

1.  Open `Intent_and_trajectory.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2.  If using Google Colab, run the first cell to mount your Google Drive and locate the dataset.
3.  Execute the cells sequentially:
    * **Step 1:** Load the nuScenes dataset and generate train/validation splits.
    * **Step 2:** Initialize the PyTorch model (`IntentAndTrajectoryPredictor`) and the custom ADE/FDE loss function.
    * **Step 3:** Run the Training Loop cell. The model will output loss and metrics per epoch.
    * **Step 4:** Run the Visualization cell to see plotted trajectory predictions.
    * **Step 5:** Run the Validation script to evaluate the model on unseen data.

## Example Outputs / Results

### Metrics Tracked
The model evaluates success using a "Best-of-N" strategy against the following metrics:
* **ADE (Average Displacement Error):** The mean Euclidean distance between the predicted points and the ground truth points across the entire 3-second prediction window.
* **FDE (Final Displacement Error):** The distance between the final predicted point (at exactly 3 seconds) and the actual final position.

### Visual Outputs
The provided visualization script generates a 2D top-down plot of the agent's coordinate frame, displaying:
* **Past Trajectory (Blue Line):** 2 seconds of history.
* **Ground Truth Future (Green Line):** The actual path taken.
* **Predicted Modes (Dashed Lines):** The top 3 predicted paths, with line thickness and opacity scaled based on the model's assigned confidence probability.
