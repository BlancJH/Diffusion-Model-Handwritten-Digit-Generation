# Diffusion-Model-Handwritten-Digit-Generation
This project implements a Denoising Diffusion Probabilistic Model (DDPM) in PyTorch to generate handwritten digits. The model learns to iteratively remove noise from random Gaussian samples, ultimately producing high-quality digit images similar to the MNIST dataset.

There are two versions of DDPM in PyTorch to generate handwritten digits:

1. Linear β-Schedule Diffusion Model

2. Cosine β-Schedule Diffusion Model

Both models follow the core DDPM framework but use different noise schedules, allowing comparison of stability, sample quality, and training behavior.

These notebooks showcase practical understanding of diffusion processes, noise schedules, training dynamics, and sampling-based generative modelling.

## Key Features

- Two complete DDPM implementations (Linear and Cosine noise schedules)

- Custom β-schedulers for forward noising

- Fully implemented forward and reverse diffusion processes

- Sampling pipeline to generate digits from pure Gaussian noise

- Side-by-side comparison of schedule performance

- Training loss visualization

- Generated sample grids for each version

- Clean, modular, and well-documented code suitable for portfolio use

## Model Variants

### 1. Linear β-Schedule (Baseline DDPM)

A classic implementation where β increases linearly over timesteps.

Demonstrating:

- Basic DDPM mechanics

- Stability

- Implementation clarity

- Simple noise schedule behavior

Notebook:

### 2. Cosine β-Schedule (Improved Signal-to-Noise Ratio)

Implements the cosine schedule introduced in Nichol & Dhariwal (Improved Denoising Diffusion Models), known for:

- Better signal preservation

- Smoother noise decay

- Higher-quality samples with fewer steps

Notebook:

## Results
### Linear Schedule Sample Outputs

### Cosine Schedule Sample Outputs

## Repository Structure
├── diffusion_model_digit_generation.ipynb         

├── diffusion_model_digit_generation(cosine schedule).ipynb

├── README.md                                             

## Technologies Used

- Python

- PyTorch

- NumPy

- Matplotlib

- MNIST Dataset

- Jupyter Notebook





