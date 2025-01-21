# Dynamic Traffic Light Control with Deep Q-Learning Algorithm

## Project Description
This project implements a dynamic traffic light control system using Deep Q-Learning (DQN) to optimize traffic flow at a four-way intersection. The system adjusts traffic light durations based on real-time data, such as vehicle counts and waiting times, in order to minimize congestion and improve traffic efficiency.

---

## Abstract
The objective of this project is to develop and implement machine learning techniques to optimize traffic flow at intersections managed by traffic light systems. Due to the legal and practical limitations of experimenting on real-world roads, a realistic traffic simulation environment was created. This simulation is integrated with live camera feeds and advanced image processing techniques to enable real-time data collection and interaction with the AI model. Using reinforcement learning, the model dynamically adjusts traffic light sequences based on real-time traffic density, aiming to minimize vehicle waiting times and enhance overall traffic flow efficiency. A distinctive feature of this model is its ability to open the same lane consecutively without a fixed cycle, responding to fluctuating traffic conditions. Additionally, unlike traditional systems relying on GPS, this system utilizes camera-based object detection for real-time vehicle detection, providing an accurate, scalable, and cost-effective solution. This approach offers a scalable solution for modern urban traffic management challenges.

---

## Core Features
1. **Reinforcement Learning:**
   - Deep Q-Learning (DQN) model for dynamic traffic light timing.
   - Rewards based on vehicle wait times, throughput, and balance between lanes.

2. **Real-Time Data Integration:**
   - Vehicle detection using YOLO (You Only Look Once).
   - Real-time updates of lane states (vehicle count, waiting times, etc.).

3. **Traffic Light Management:**
   - Dynamic adjustment of green light durations for each lane (15-30 seconds).
   - Coordination between opposite lanes (e.g., 1-3, 2-4).

4. **Simulation Environment:**
   - Custom simulation software for modeling traffic flow.
   - Real-time synchronization with external data sources for vehicle detection.

---

## Technical Stack
- **Backend:** Python with TensorFlow/Keras for DQN implementation
- **Simulation:** Custom-built software for traffic flow simulation
- **Integration:** YOLO for vehicle detection

---

## Expected Outcomes
- A fully functional DQN-based traffic light system that adapts to changing traffic conditions.
- Improved traffic flow and reduced waiting times at intersections, leading to optimized urban traffic management.

---

## Project Screenshots

### Traffic Flow Simulation:
![Traffic Flow](https://github.com/user-attachments/assets/eee330d8-09bb-4172-8a7a-8ee446c51cc9)

### Real-time Vehicle Detection:
![Vehicle Detection](https://github.com/user-attachments/assets/b5d5c77c-2a91-445c-b2a3-79bf8870e059)

### Traffic Light Management in Action:
![Traffic Lights](https://github.com/user-attachments/assets/02f51a79-be6d-4f7d-8647-b8e3d630f57a)

### Simulation Results:
![Simulation Results](https://github.com/user-attachments/assets/ce7f8a21-be78-42b9-baaf-98a640492c38)

---
