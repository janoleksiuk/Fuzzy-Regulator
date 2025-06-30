# Inverted Pendulum Simulation with Fuzzy Logic Control

This project simulates an inverted pendulum mounted on cart (along 1 axis) system using a fuzzy logic controller. It includes a real-time graphical interface for visualization, built with PyQtGraph, and a plotting window to monitor the control signal over time.

## Features

- Real-time simulation of an inverted pendulum on a cart
- Fuzzy logic controller for dynamic stabilization
- Configurable system parameters and disturbances
- Visualization using PyQtGraph
- Live plotting of control force with Matplotlib

## Example Output

![Zrzut ekranu 2025-06-30 184358](https://github.com/user-attachments/assets/2773012b-cb61-40ca-9317-38343565284b)

## Custom Initial Conditions

To set your own initial conditions, **modify the values in lines 430 and 431** of the `inverted_pendulum.py` file:

```python
pendulum = InvertedPendulum(
    x0=0, dx0=0, theta0=np.pi/10, dtheta0=0, 
    ih=800, iw=1000, h_min=-80, h_max=80
)
```

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
