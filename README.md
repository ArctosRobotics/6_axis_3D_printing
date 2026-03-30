# Arctos Studio 2.6 - 3D Printing & Robotic Automation

Arctos Studio is a powerful robotic control and simulation environment designed for Arctos robotic arms. This repository specifically focuses on the integration of **3D printing** capabilities with robotic arms, enabling multi-axis printing and automated production workflows.

## 🚀 Key Features

- **Integrated STL Slicer**: A built-in geometric engine that converts STL models into 2D toolpaths optimized for robotic motion.
- **3D Robot Printing**: Move beyond traditional 3D printers by using a 6-axis robotic arm for complex printing paths.
- **Bambu Lab Integration**: Real-time monitoring and automation for Bambu Lab 3D printers via MQTT.
- **Automated Part Removal**: Example scripts to coordinate the robotic arm and 3D printer for "lights-out" manufacturing.
- **Syringe Extruder Support**: Compatible with custom syringe-based extruders for advanced material deposition (see `syringe_extruder/` folder for STL components).

## 🛠 Project Structure

- **`printing_3d.py`**: The main interface for the 3D printing module.
- **`stl_slicer.py`**: Core geometry engine responsible for slicing meshes into layers.
- **`stl_handler.py`**: Manages STL model loading, positioning, and scaling.
- **`bambu_handler.py`**: Service for MQTT communication with Bambu Lab printers.
- **`bambu_panel.py`**: Dashboard for printer monitoring.
- **`syringe_extruder/`**: (Referenced) Folder containing STL files for the custom syringe extruder hardware.
- **`gcode_examples/`**: Sample G-code and motion files for testing.

## 📦 Installation

To use the 3D printing module, ensure you have the following dependencies installed:

```bash
pip install trimesh shapely scipy paho-mqtt numpy PyQt5
```

For full UI support, it is recommended to install all requirements:
```bash
pip install -r requirements.txt
```

## 🏎 Quick Start

1. **Test Motion**: Run `generate_test_cube_gcode.py` to create a sample 20mm cube G-code.
2. **Setup Extruder**: Build the syringe extruder using the STL files in the `syringe_extruder/` folder.
3. **Connect & Slice**:
   - Open Arctos Studio.
   - Load an STL model.
   - Access the **3D Printing Slicer** via the UI (powered by `printing_3d.py`).
   - Configure your layer and speed settings.
   - Generate robot targets and start printing!

## 🤖 Automated Workflows

You can find automation examples like `example_bambu_automation.py` which demonstrates how to wait for a print job to finish and then use the robot to automatically clear the build plate.

---
*Developed by Arctos Studio Team.*
