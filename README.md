# Dynamic-Traffic-Light-Control-with-Deep-Q-Learning-Algorithm

#### Project Description:
This project implements a traffic light control system using Deep Q-Learning (DQN) to optimize traffic flow at a four-way intersection. The system dynamically adjusts traffic light durations based on real-time data, such as vehicle counts and waiting times, to minimize congestion and improve efficiency.

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






![image](https://github.com/user-attachments/assets/437208e3-3ff6-4edb-a3c7-067cd1b24c61)

![image](https://github.com/user-attachments/assets/e18d3b92-fffd-49fb-a58f-e533fe0edd87)

