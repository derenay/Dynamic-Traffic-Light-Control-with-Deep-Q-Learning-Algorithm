    import json
import threading
import time
from datetime import datetime, timedelta
from multiprocessing import Process, Queue
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import logging
import os


# Loglama ayarları
logging.basicConfig(filename='dqn_agent_log.json', level=logging.INFO, format='%(message)s')

logging.basicConfig(filename='state_log.json', level=logging.INFO, format='%(message)s')
# Trafik Simülasyonu ve Araç Algılama Kısmı...

# Şerit poligonları (Fotoğraflardan tanımlanan şeritler)
lane_polygons = {
    # KARATAY MEDRESESİ
    1: np.array([(128, 993), (933, 991), (855, 362), (641, 348)], np.int32),  # Sol
    3: np.array([(935, 988), (1824, 955), (1054, 364), (853, 362)], np.int32), # Sağ

    # YENİ MERAM CADDESİ
    2: np.array([(817, 406), (692, 988), (29, 962), (704, 396)], np.int32),    # Sol
    4: np.array([(721, 982), (815, 408), (980, 396), (1728, 985)], np.int32)   # Sağ
}

# YOLO modeli yolu
MODEL_PATH = r"C:\\Users\\erena\\Desktop\\Yolo\\carDetectionTrain\\model\\yolo11m-seg.pt"

# İzin verilen araç sınıfları
ALLOWED_CLASSES = ['car', 'bus', 'motorcycle', 'truck']

def vehicle_detection_process(queue, video_url, camera_index, lanes):
    """
    Belirli bir kameradan araç tespiti yapar ve sonuçları Queue'ya gönderir.

    Args:
        queue (Queue): Sonuçların gönderileceği Queue.
        video_url (str): Kameranın video akış URL'si.
        camera_index (int): Kameranın index numarası.
        lanes (list): Kameraya atanmış şerit numaraları.
    """
    model = YOLO(MODEL_PATH).cuda()
    unique_track_ids = {lane: set() for lane in lanes}  # Şeritler için benzersiz ID'ler

    while True:
        try:
            cap = cv2.VideoCapture(video_url)
            if not cap.isOpened():
                print(f"Kamera {camera_index} - Video akışı açılamadı.")
                time.sleep(2)
                continue

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print(f"Kamera {camera_index} - Frame alınamadı.")
                    time.sleep(2)
                    break

                # YOLO modelini çalıştır
                results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

                # Eğer sonuçlar geçerli değilse devam et
                if not results or results[0] is None or results[0].boxes.id is None:
                    continue

                # Track edilen araçları işle
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                vehicle_types = results[0].boxes.cls.cpu().numpy()
                names = results[0].names

                for box, track_id, vehicle_type in zip(boxes, track_ids, vehicle_types):
                    class_label = names[int(vehicle_type)]
                    if class_label not in ALLOWED_CLASSES:
                        continue

                    # Araç merkez noktasını hesapla
                    center_x, center_y = int(box[0]), int(box[1])

                    # Her bir şeridi kontrol et
                    for lane_id in lanes:
                        polygon = lane_polygons[lane_id]
                        if cv2.pointPolygonTest(polygon, (center_x, center_y), False) >= 0:
                            if track_id not in unique_track_ids[lane_id]:
                                unique_track_ids[lane_id].add(track_id)

                                # Queue'ya bilgi gönder
                                queue.put({
                                    "camera_index": lane_id,
                                    "lane_id": lane_id,
                                    "track_id": int(track_id),
                                    "class_label": class_label,
                                    "timestamp": datetime.now().isoformat()
                                })

            cap.release()
            print(f"Kamera {camera_index} - Video akışı tekrar denenecek.")
            time.sleep(2)

        except Exception as e:
            print(f"Kamera {camera_index} - Bir hata oluştu: {e}. Tekrar denenecek.")
            time.sleep(2)


class Vehicle:
    vehicle_counter = 0  # Benzersiz ID için sayaç

    def __init__(self, class_label, entry_time):
        Vehicle.vehicle_counter += 1
        self.id = Vehicle.vehicle_counter  # Her araç için benzersiz bir ID
        self.class_label = class_label  # Araç tipi (örneğin car, bus, truck)
        self.entry_time = entry_time  # Şeride giriş zamanı
        self.wait_time = 0  # Kırmızı ışıkta bekleme süresi
        self.pass_time = None  # Yeşil ışıkta geçiş zamanı
        self.total_time = None  # Şeritte toplam kaldığı süre

    def calculate_wait_time(self, current_time):
        """Araç kırmızı ışıkta bekliyorsa bekleme süresini hesaplar"""
        self.wait_time = current_time - self.entry_time  # Kırmızı ışıkta bekleme süresi

    def calculate_pass_time(self, pass_time):
        """Araç yeşil ışıkta geçtiğinde geçiş süresi hesaplanır"""
        self.pass_time = pass_time  # Araç geçiş zamanı
        self.total_time = self.pass_time - self.entry_time  # Toplam şeritte kaldığı süre

    def __repr__(self):
        return f"Vehicle(ID={self.id}, class={self.class_label}, entry_time={self.entry_time}, wait_time={self.wait_time}, pass_time={self.pass_time}, total_time={self.total_time})"


lane_vehicle_counts = {1: 0, 2: 0, 3: 0, 4: 0}
total_vehicle_count = {1: 0, 2: 0, 3: 0, 4: 0}
vehicles_in_lane = {1: [], 2: [], 3: [], 4: []}
light_states = {1: False, 2: False, 3: False, 4: False}
vehicle_exit_rate = {1: 2, 2: 2, 3: 2, 4: 2}

def control_lights(lane_to_open, green_duration):
    global light_states
    print(f"control_lights çalıştı - Şerit {lane_to_open} açık, süre: {green_duration}")

    # Bütün ışıkları kapat
    light_states[1] = False
    light_states[2] = False
    light_states[3] = False
    light_states[4] = False

    # Ajanın aksiyonuna göre ilgili şeritleri aç
    if lane_to_open == 1:
        light_states[1] = True
        light_states[3] = True
        print(f"1 ve 3. şeritler {green_duration} saniye boyunca açık.")
    elif lane_to_open == 2:
        light_states[2] = True
        light_states[4] = True
        print(f"2 ve 4. şeritler {green_duration} saniye boyunca açık.")

    # Yeşil ışık süresini beklet
    time.sleep(green_duration)

def manage_vehicle_exits():
    global lane_vehicle_counts, total_vehicle_count, vehicles_in_lane

    while True:
        for lane_index in range(1, 5):
            if light_states[lane_index]:
                exiting_vehicles = min(vehicle_exit_rate[lane_index], lane_vehicle_counts[lane_index])

                for _ in range(exiting_vehicles):
                    if vehicles_in_lane[lane_index]:
                        vehicle = vehicles_in_lane[lane_index].pop(0)
                        vehicle.calculate_pass_time(time.time())
                        total_vehicle_count[lane_index] += 1

                lane_vehicle_counts[lane_index] -= exiting_vehicles

        time.sleep(1)

def update_wait_times():
    global vehicles_in_lane

    while True:
        for lane_index in range(1, 5):
            if not light_states[lane_index]:
                current_time = time.time()
                entry_times = np.array([vehicle.entry_time for vehicle in vehicles_in_lane[lane_index]])

                wait_times = current_time - entry_times
                for i, vehicle in enumerate(vehicles_in_lane[lane_index]):
                    vehicle.wait_time = wait_times[i]

        time.sleep(1)

def display_lane_status():
    while True:
        status = "\n--- Şerit Durumları ---\n"
        lane_status = {}  # Dictionary to store the lane status

        for lane in range(1, 5):
            # Create a dictionary entry for each lane
            lane_status[lane] = {
                'light_state': 'Yeşil' if light_states[lane] else 'Kırmızı',
                'vehicle_count': lane_vehicle_counts[lane],
                'total_passed': total_vehicle_count[lane]
            }

            # Update the printed status string
            status += (f"{lane}. şerit: {lane_status[lane]['light_state']}, "
                       f"Mevcut araç sayısı: {lane_status[lane]['vehicle_count']}, "
                       f"Toplam geçen araç sayısı: {lane_status[lane]['total_passed']}\n")

        print(status)  # Print the status
        time.sleep(1)  # Sleep for 1 second between updates

def update_lane_vehicle_count(detected_vehicles):
    global lane_vehicle_counts, vehicles_in_lane
    for vehicle_data in detected_vehicles:
        lane_index = vehicle_data["camera_index"]
        lane_vehicle_counts[lane_index] += 1

        new_vehicle = Vehicle(class_label=vehicle_data["class_label"], entry_time=time.time())
        vehicles_in_lane[lane_index].append(new_vehicle)

def handle_queue_data(queue):
    while True:
        if not queue.empty():
            vehicle_data = queue.get()
            update_lane_vehicle_count([vehicle_data])



class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_lane = nn.Linear(128, 2)      # Şerit seçimi için çıktı (2 değer)
        self.fc_duration = nn.Linear(128, 16)  # Süre seçimi için çıktı (15-30 arası 16 değer)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        lane_output = self.fc_lane(x)         # Şerit seçimi için çıktı
        duration_output = self.fc_duration(x) # Süre seçimi için çıktı
        return lane_output, duration_output


class DQNAgent:
    def __init__(self, state_dim, lr, gamma, epsilon, epsilon_decay, buffer_size):
        self.state_dim = state_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=buffer_size)
        self.model = DQN(state_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Günün her saati için trafik yoğunlukları (24 saat diliminde her şerit için)
        self.time_based_intensity = {hour: {1: 0, 2: 0, 3: 0, 4: 0} for hour in range(24)}

    def update_intensity(self, current_hour, lane_vehicle_counts):
        """Her saat diliminde şeritlerdeki araç yoğunluğunu güncelle"""
        for lane in range(1, 5):
            self.time_based_intensity[current_hour][lane] += lane_vehicle_counts[lane]

    def calculate_waiting_time(self, vehicles_in_lane):
        """Şeritteki her aracın bekleme süresini hesapla ve ortalamasını al"""
        current_time = time.time()
        waiting_times = [current_time - vehicle.entry_time for vehicle in vehicles_in_lane]
        if waiting_times:
            return np.mean(waiting_times)
        return 0

    def reward_function(self, state, next_state):
        """
        Reward function to evaluate the effectiveness of the selected action.

        Args:
        - state: The current state of the traffic system.
        - next_state: The state of the traffic system after applying an action.

        Returns:
        - reward: A calculated reward value based on the state transition.
        """
        # Identify which lanes are opened based on the next_state light configuration
        lanes_opened = [1, 3] if next_state[1] == 1 else [2, 4]
        other_lanes = [lane for lane in range(1, 5) if lane not in lanes_opened]

        # Calculate densities in the current and next states
        opened_density = sum(state[(lane - 1) * 3] for lane in lanes_opened)
        other_density = sum(state[(lane - 1) * 3] for lane in other_lanes)
        opened_density_next = sum(next_state[(lane - 1) * 3] for lane in lanes_opened)
        other_density_next = sum(next_state[(lane - 1) * 3] for lane in other_lanes)

        # Initialize reward components
        reward = 0
        density_penalty = 0
        flow_reward = 0
        flow_penalty = 0

        # Reward for reducing the density in the opened lanes
        if opened_density > opened_density_next:
            flow_reward += (opened_density - opened_density_next) * 2

        # Penalty for increasing density in other lanes
        if other_density_next > other_density:
            density_penalty -= (other_density_next - other_density) * 3

        # Penalty for imbalance between opened and other lanes
        if opened_density_next == 0 and other_density_next > 0:
            density_penalty -= other_density_next * 2

        # Additional reward if opened lanes clear a significant portion of their vehicles
        if opened_density_next / (opened_density + 1e-5) < 0.5:  # Avoid division by zero
            flow_reward += 10

        # Large penalty if other lanes' density exceeds a threshold
        for lane in other_lanes:
            lane_density_next = next_state[(lane - 1) * 3]
            if lane_density_next > 20:
                density_penalty -= lane_density_next * 2

        if opened_density == 0 and other_density > 0:
            flow_penalty = -50

        # Combine reward components
        reward = flow_reward + density_penalty + flow_penalty

        return reward



    def act(self, state, tau=0.7):
        """Boltzmann Exploration stratejisi ile aksiyon seç"""
        lane_output, duration_output = self.model(torch.tensor(state, dtype=torch.float32))

        # Şerit seçim olasılıkları
        lane_probabilities = torch.softmax(lane_output / tau, dim=0).detach().numpy()
        lane_action = np.random.choice([1, 2], p=lane_probabilities)
        print(f"lane_probabilities: {lane_probabilities}, lane_action: {lane_action}")
        # Süre seçim olasılıkları (15-30 saniye aralığında)
        duration_probabilities = torch.softmax(duration_output / tau, dim=0).detach().numpy()
        duration_action = np.random.choice(np.arange(15, 31), p=duration_probabilities)

        # Aksiyon listesi
        action = [lane_action, int(duration_action)]

        # Günün saatiyle ilişkili yoğunluğu kaydediyoruz
        current_hour = int(datetime.now().hour)
        self.update_intensity(current_hour, lane_vehicle_counts)

        return action

    def remember(self, state, action, next_state, done):
        reward = self.reward_function(state, next_state)  # Ödül hesaplama
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Geçmiş deneyimleri öğrenmek ve model güncellemek için"""
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            
            # Hedefi hesapla
            target = reward
            if not done:
                next_q_values = self.model(torch.tensor(next_state, dtype=torch.float32))
                lane_next_q, duration_next_q = next_q_values
                target = reward + self.gamma * max(torch.max(lane_next_q).item(), torch.max(duration_next_q).item())

            # Şu anki durumun Q değerlerini al
            current_q_values = self.model(torch.tensor(state, dtype=torch.float32))
            lane_q_values, duration_q_values = current_q_values
            target_f_lane = lane_q_values.clone().detach()
            target_f_duration = duration_q_values.clone().detach()

            # Eylem için hedef değeri ayarla
            target_f_lane[action[0] - 1] = target  # Şerit seçimi için
            target_f_duration[action[1] - 15] = target  # Süre seçimi için (15-30 aralığında indeksleme)

            # Modeli eğit
            self.optimizer.zero_grad()
            loss_lane = nn.MSELoss()(lane_q_values, target_f_lane)
            loss_duration = nn.MSELoss()(duration_q_values, target_f_duration)
            loss = loss_lane + loss_duration
            loss.backward()
            self.optimizer.step()

        # Epsilon'u düşürerek daha az rastgele seçim yapılmasını sağla
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """Modeli belirtilen dosya adına kaydeder"""
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """Modeli belirtilen dosyadan geri yükler"""
        self.model.load_state_dict(torch.load(filename))
        

def get_state():
    state = []  # Durum vektörü

    current_time = datetime.now()  # Şu anki zaman
    log_data = {"timestamp": current_time.isoformat(), "state_info": []}  # Log için veri hazırlığı

    for i in range(1, 5):
        # Şeritteki toplam araç sayısı (anlık araç sayısı)
        vehicle_count = lane_vehicle_counts[i]
        state.append(vehicle_count)  # Şeritteki araç sayısını listeye ekle
        
        # Işık durumu (1: yeşil, 0: kırmızı)
        light_state = int(light_states[i])
        state.append(light_state)  # Işık durumunu listeye ekle

        # Ortalama bekleme süresini hesapla
        if vehicle_count > 0:
            avg_waiting_time = np.mean([vehicle.wait_time for vehicle in vehicles_in_lane[i]])
        else:
            avg_waiting_time = 0  # Şeritte araç yoksa bekleme süresi 0
        
        state.append(avg_waiting_time)  # Ortalama bekleme süresini listeye ekle

        # Log verisine bu şerit için bilgileri ekle
        log_data["state_info"].append({
            "lane": i,
            "vehicle_count": vehicle_count,
            "light_state": light_state,
            "avg_waiting_time": avg_waiting_time,
            "total_vehicle_count": total_vehicle_count[i]
        })
    
    print(state)  # Listeyi yazdır
    
    # State verilerini logla (JSON formatında)
    logging.info(json.dumps(log_data))
    
    return state



def manage_traffic_lights_with_agent(agent):
    start_time = time.time()  # Başlangıç zamanını kaydet
    model_save_dir = 'models/'  # Modelin kaydedileceği klasör

    # Eğer klasör yoksa oluştur
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    while True:
        state = get_state()
        action = agent.act(state)  # Aksiyon al
        print(f"DQN agent aksiyonu: {action}")

        # Ajanın aksiyonuna göre hangi şeritlerin açılacağını ve süresini kontrol et
        lane_to_open = action[0]  # 1: 1-3 şeritler, 2: 2-4 şeritler
        green_duration = action[1]  # Yeşil ışık süresi
        control_lights(lane_to_open, green_duration)

        # Her 10 dakikada bir modeli kaydet
        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time >= 600:  # 600 saniye = 10 dakika
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(model_save_dir, f'dqn_agent_model_{timestamp}.pth')
            agent.save(model_path)
            print(f"Model {model_path} zamanında kaydedildi.")
            start_time = current_time  # Zamanlayıcıyı sıfırla


def main(queue):
    # DQN ajanı yaratma
    agent = DQNAgent(state_dim=12, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, buffer_size=2000)
    
    model_path = r'C:\Users\erena\Desktop\Yolo\models\dqn_agent_model_20241204_132256.pth' #C:\Users\erena\Desktop\Yolo\models\dqn_agent_model_20241121_161319.pth
    try:
        agent.load(model_path)
        print("Model başarıyla yüklendi.")
    except FileNotFoundError:
        print("Model dosyası bulunamadı, yeni modelle başlanacak.")

    # Diğer thread'leri başlatma...
    traffic_light_thread = threading.Thread(target=manage_traffic_lights_with_agent, args=(agent,))
    traffic_light_thread.start()

    vehicle_exit_thread = threading.Thread(target=manage_vehicle_exits)
    vehicle_exit_thread.start()

    wait_time_thread = threading.Thread(target=update_wait_times)
    wait_time_thread.start()

    display_status_thread = threading.Thread(target=display_lane_status)
    display_status_thread.start()

    queue_data_thread = threading.Thread(target=handle_queue_data, args=(queue,))
    queue_data_thread.start()

if __name__ == "__main__":
    vehicle_queue = Queue(maxsize=1000)

    # Kameralar için URL'ler
    camera_1_url = "https://content.tvkur.com/l/c77i5e384cnrb6mlji10/master.m3u8"  # KARATAY MEDRESESİ
    camera_2_url = "https://content.tvkur.com/l/c77i6cfbb2nj4i0fr7s0/master.m3u8"  # YENİ MERAM CADDESİ

    # Process başlat
    processes = [
        Process(target=vehicle_detection_process, args=(vehicle_queue, camera_1_url, 1, [1, 3])),
        Process(target=vehicle_detection_process, args=(vehicle_queue, camera_2_url, 2, [2, 4]))
    ]

    for process in processes:
        process.start()

    p2 = Process(target=main, args=(vehicle_queue,))
    p2.start()

    for process in processes:
        process.join()

    p2.join()
