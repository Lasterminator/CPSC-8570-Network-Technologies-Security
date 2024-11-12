# A Novel Approach to Guard from Adversarial Attacks using Stable Diffusion

## Project Overview
This project presents a dynamic and adaptive strategy for defending AI systems against adversarial attacks. By leveraging stable diffusion techniques, we aim to enhance the robustness and adaptability of AI models, providing a comprehensive defense against a wide range of adversarial threats.

## Data Description
Our project utilizes datasets from various domains, including image classification and face recognition, to evaluate the effectiveness of our proposed defense mechanism. The datasets include MNIST, GTSRB, VGG-Face, and Youtube-Face. These datasets have been preprocessed to facilitate efficient training and evaluation of our models.

## Methodology
- **Stable Diffusion Training:** Training models using a stable diffusion algorithm to create resilient AI systems.
- **Attack Simulation:** Implementing both white box and black box adversarial attacks, including Projected Gradient Descent (PGD) and Fast Gradient Sign Method (FGSM).
- **Defense Mechanism:** Applying stable diffusion techniques to mitigate the impact of adversarial attacks and improve model robustness.

## Results
- **White Box Attack - PGD:**
  - Success rate without diffusion: 90.8%
  - Success rate with diffusion: 4.2%
- **White Box Attack - FGSM:**
  - Success rate without diffusion: 71.7%
  - Success rate with diffusion: 8.8%
- **Black Box Attack - PGD:**
  - Success rate without diffusion: 55.8%
  - Success rate with diffusion: 2.7%
- **Black Box Attack - FGSM:**
  - Success rate without diffusion: 65.8%
  - Success rate with diffusion: 5.6%

The results demonstrate a significant reduction in the success rates of adversarial attacks when stable diffusion techniques are applied, highlighting their effectiveness in enhancing AI system security.

## Contributors
- **Uma Maheswara Rao Meleti** - umeleti@g.clemson.edu
- **Trinath Sai Subhash Reddy Pittala** - tpittal@g.clemson.edu
- **Geethakrishna Puligundla** - gpuligu@g.clemson.edu
