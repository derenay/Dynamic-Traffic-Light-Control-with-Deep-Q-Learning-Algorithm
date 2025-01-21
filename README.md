# Dynamic-Traffic-Light-Control-with-Deep-Q-Learning-Algorithm

#### Project Description:
This project implements a traffic light control system using Deep Q-Learning (DQN) to optimize traffic flow at a four-way intersection. The system dynamically adjusts traffic light durations based on real-time data, such as vehicle counts and waiting times, to minimize congestion and improve efficiency.


#### Abstract

The objective of this project is to develop and implement machine learning techniques to optimize traffic flow at intersections managed by traffic light systems. Due to the legal and practical limitations of experimenting on real-world roads, a realistic traffic simulation environment was created. This simulation is integrated with live camera feeds and advanced image processing to enable real-time data collection and interaction with the AI model. Using reinforcement learning, the model adapts traffic light sequences based on real-time traffic density, aiming to minimize vehicle waiting times and enhance overall traffic flow efficiency. A distinctive feature of this model is its ability to dynamically open the same lane consecutively without a fixed cycle, responding to fluctuating traffic conditions. Additionally, unlike traditional systems relying on GPS, the system utilizes camera-based object detection for real-time vehicle detection, providing an accurate, scalable, and cost-effective solution. This approach offers a scalable solution for modern urban traffic management challenges.
---

#### Core Features:
1. **Reinforcement Learning:**
   - Deep Q-Learning model to decide traffic light timings.
   - Rewards based on vehicle wait times, throughput, and balance between lanes.

2. **Real-Time Data Integration:**
   - Vehicle detection using YOLO.
   - Real-time updates of lane states (vehicle count, waiting time).

3. **Traffic Light Management:**
   - Dynamic adjustment of green light durations for each lane.
   - Coordination between opposite lanes (e.g., 1-3, 2-4).
   - the traffic light times between 15-30 seconds.

4. **Simulation Environment:**
   - Integration with custom made similation envoriment.
   - Real-time synchronization with external data.

---

#### Technical Stack:
- **Backend:** Python with TensorFlow/Keras for DQN implementation
- **Simulation:** Special(Developing) software for traffic flow simulation
- **Integration:** YOLO for vehicle detection

---

#### Expected Outcomes:
- A fully functional DQN-based traffic light system that adapts to changing traffic patterns.
- Improved traffic flow and reduced waiting times at intersections.

---




![image](https://github.com/user-attachments/assets/eee330d8-09bb-4172-8a7a-8ee446c51cc9)
![image](https://github.com/user-attachments/assets/b5d5c77c-2a91-445c-b2a3-79bf8870e059)
![image](https://github.com/user-attachments/assets/02f51a79-be6d-4f7d-8647-b8e3d630f57a)
![image](https://github.com/user-attachments/assets/ce7f8a21-be78-42b9-baaf-98a640492c38)



![image](https://github.com/user-attachments/assets/437208e3-3ff6-4edb-a3c7-067cd1b24c61)

![image](https://github.com/user-attachments/assets/e18d3b92-fffd-49fb-a58f-e533fe0edd87)

