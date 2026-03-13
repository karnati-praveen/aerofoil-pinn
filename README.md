# Physics-Informed Neural Network (PINN) for Aerofoil Flow

Predicts **velocity (u, v)** and **pressure (p)** fields around an aerofoil using PINNs trained on ANSYS Fluent CFD simulation data.

## What It Does
- Trains 3 separate neural networks for velocity-u, velocity-v, and pressure
- Embeds **Navier-Stokes equations** directly into the loss function (no labelled physics data needed)
- Dataset: 3,668 spatial points exported from ANSYS Fluent aerofoil simulation

## Loss Function
```
Total Loss = 10 × Data Loss + 0.001 × Physics Loss + 10 × Boundary Loss
```
- **Data Loss** — MSE against ANSYS ground truth  
- **Physics Loss** — continuity + x/y momentum residuals (Navier-Stokes)  
- **Boundary Loss** — no-slip wall conditions  

## Network Architecture
```
Input(x, y) → Linear(200) → Tanh → Linear(200) → Tanh → Linear(1)
```
Three identical networks trained independently for u, v, p.

## Tech Stack
`Python` `PyTorch` `Pandas` `Matplotlib` `Scientific ML` `Navier-Stokes`

## Setup & Run
```bash
pip install -r requirements.txt
jupyter notebook aerofoil_pinn.ipynb
```

## Project Structure
```
aerofoil-pinn/
├── aerofoil_pinn.ipynb      ← main notebook
├── pinn_aerofoil_data.csv   ← ANSYS CFD dataset
├── requirements.txt
└── README.md
```

## Author
Karnati Praveen — B.Tech AI & Data Science, Amrita Vishwa Vidyapeetham
