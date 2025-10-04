# Flood Segmentation with U-Net

This repository implements a **U-Net–based deep learning model** for detecting flooded regions in satellite imagery. Flooding is one of the most destructive natural disasters globally, especially in regions where disaster management resources are limited. Accurate and automated flood detection is critical for:

* **Early warning systems**
* **Resource allocation during crises**
* **Post-disaster damage assessment**

Our solution leverages **semantic segmentation** to classify each pixel in an image as “flooded” or “non-flooded.” The U-Net architecture is particularly well-suited to this task due to its encoder–decoder structure and ability to preserve spatial information while capturing high-level context.

---

##  Project Structure

```bash
.
├── best_unet.pth          # Saved model weights (best checkpoint)
├── data/                  # Dataset management
│   ├── dataset.py
│   └── __pycache__/
├── inference.py           # Run inference on new satellite images
├── model.py               # U-Net model definition
├── notebooks/
│   └── demo.ipynb         # Visualization and analysis
├── __pycache__/
│   ├── model.cpython-312.pyc
│   └── utils.cpython-312.pyc
├── README.md
├── requirements.txt       # Dependencies
├── test/                  # Automated tests
│   ├── sanity_check.py
│   └── test_dataset.py
├── train.py               # Training pipeline
└── utils.py               # Utility functions (metrics, transforms)
```

---

## How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/flood-segmentation.git
   cd flood-segmentation
   ```

2. **Set up environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Train**

   ```bash
   python train.py
   ```

   The best model is saved to `best_unet.pth`.

4. **Inference**

   ```bash
   python inference.py --image path/to/test_image.png
   ```

5. **Visualization**
   Open the notebook:

   ```bash
   jupyter notebook notebooks/demo.ipynb
   ```

---

##  Testing

i included simple automated tests to ensure reliability:

* `sanity_check.py`: validates dataset integrity and transforms.
* `test_dataset.py`: verifies dataset loading and structure.

Run:

```bash
pytest test/
```

---

## Results (Sample)

After training, we visualize predictions using the demo notebook. Outputs include:

* Original satellite image
* Ground truth mask
* Model-predicted mask

*(Plots will be added once sample runs are included.)*

---

## Future Work

* Incorporating **attention mechanisms** to improve segmentation accuracy.
* Experimenting with **larger datasets** for better generalization.
* Deploying as an **API service** for real-time disaster management.

---

## Why This Project?

Flooding has direct social and economic consequences. A robust, automated flood segmentation model can support governments, NGOs, and researchers in making data-driven decisions during disaster response.

This project demonstrates:

* **Strong technical foundations** in computer vision.
* **Practical relevance** to real-world challenges.
* A **scalable baseline** for future extensions (more advanced architectures, multi-class segmentation, deployment pipelines).

---
