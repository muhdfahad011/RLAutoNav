# RLAutoNav
A self-driving AI system that uses a custom CNN model trained with reinforcement learning concepts to map visual input to driving actions. Designed for real-time decision-making in simulated environments, enabling end-to-end autonomous vehicle control.


Here's a well-written `README.md` description for your self-driving AI project, including your requested improvements and highlighting the synchronization between both files:

---

# 🚗 Self-Driving AI using CARLA Simulator

This repository contains a deep reinforcement learning agent designed to learn autonomous driving behavior within the CARLA simulation environment. It uses a custom environment wrapper and a convolutional neural network to process grayscale camera input and output steering decisions.

## 🧠 Project Overview

The project uses a **Discrete Deep Q-Network (DQN)** to train a car to drive autonomously by interpreting front-facing camera images and choosing from three discrete actions: **turn left**, **go straight**, or **turn right**.

The simulation environment is built on top of CARLA and integrates directly with a TensorFlow-based learning loop.

---

## ⚙️ Files Included

### `car_environment.py`

* Wraps the CARLA simulator into a custom OpenAI Gym-like environment.
* Includes:

  * Collision sensor handling
  * RGB camera with real-time image preprocessing
  * Speed-based reward system
  * Cleaned-up actor management
* Images are converted to grayscale and downsampled for memory and speed efficiency.

### `train_model.py`

* Builds a CNN-based DQN agent using TensorFlow.
* Uses experience replay and a target network to stabilize learning.
* Includes:

  * Dynamic reward shaping
  * Early synchronization with the camera sensor
  * Efficient replay memory updates
  * Epsilon-greedy exploration
* Synchronization with the environment is improved to ensure clean image collection and proper episode resets.

---

## ✨ Improvements Made

* Separated and cleaned camera preprocessing and sensor logic.
* Reduced model input size to improve training speed.
* Enhanced state synchronization between training and environment reset.
* Modular reward system based on velocity and collisions.
* Improved readability, modularity, and training stability.

---

## 🧪 Getting Started

1. **Install CARLA simulator** (tested on CARLA 0.9.13+).
2. Clone this repo and ensure dependencies:

   ```bash
   pip install tensorflow numpy opencv-python
   ```
3. Run training:

   ```bash
   python train_model.py
   ```

---

## 📌 Requirements

* Python 3.7+
* CARLA Simulator
* TensorFlow 2.x
* NumPy, OpenCV

---

## 🤝 Contribution

Feel free to open issues or submit pull requests. Contributions to improve the training loop, reward functions, or agent behavior are welcome!

---

Let me know if you'd like badges, images (e.g., from training), a license section, or a GitHub Actions CI badge.
