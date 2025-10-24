# Obstacle-Detection-and-Sonification-of-Visual-COMS4731W

## Overview

This project implements a system for **Visual Obstacle Detection and Sonification** (VODS). The primary goal is to process visual data (RGB images and Depth maps) from the environment to detect obstacles and translate their spatial information into audible feedback (sonification), aiding visually impaired users or autonomous navigation systems.

## Setup

### 1. Requirements

- This project requires Python 3.8+.

### 2. Automated Setup

The project supports automated setup using the provided script.

```bash
# Give proper execution permit (Linux/macOS)
chmod +x setup.sh

# Run the setup script
./setup.sh
```

- This will install all dependencies into the virtual environment (`cvproject`).

#### Manual setup

- To setup virtual environment manually, check file `depedencies.txt` in a root folder to find all dependencies.

### 3. Environment Activation

You must activate the virtual environment before running any Python scripts:

| OS / Shell | Command |
| :--- | :--- |
| **Linux / macOS (Bash)** | `source cvproject/bin/activate` |
| **Windows (CMD/PowerShell)** | `.\cvproject\Scripts\activate` |
