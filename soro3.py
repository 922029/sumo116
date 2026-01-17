import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

# ==========================================
# 設定と定数
# ==========================================
SUMO_CMD = ["sumo", "-c", "4cross.sumocfg", "--no-warnings", "--no-step-log"]
# SUMO_CMD = ["sumo-gui", "-c", "4cross.sumocfg", "--no-warnings"] # GUIで見たい場合

EPISODE_LENGTH = 9000
CYCLE_TIME = 200
STEPS_PER_EPISODE = EPISODE_LENGTH // CYCLE_TIME
TOTAL_EPISODES = 100
TOTAL_TIMESTEPS = STEPS_PER_EPISODE * TOTAL_EPISODES

# 交差点定義
INTERSECTIONS = ["C1", "C2", "C3", "C4"]
NEIGHBORS = {
    "C1": ["C2", "C3"],
    "C2": ["C1", "C4"],
    "C3": ["C1", "C4"],
    "C4": ["C2", "C3"]
}

# 流入エッジの定義 (各交差点に対応する流入ID)
INCOMING_EDGES = {
    "C1": ["N1_C1", "W1_C1", "C2_C1", "C3_C1"],
    "C2": ["N2_C2", "E1_C2", "C1_C2", "C4_C2"],
    "C3": ["S1_C3", "W2_C3", "C1_C3", "C4_C3"],
    "C4": ["S2_C4", "E2_C4", "C2_C4", "C3_C4"]
}

# 行動の定義 (青時間の配分パターン)
MIN_GREEN = 20
TOTAL_GREEN_AVAILABLE = 180
NUM_GREEN_PHASES = 4 # 南北直進, 南北右折, 東西直進, 東西右折

# ==========================================
# ヘルパー関数
# ==========================================
def get_waiting_stats(intersection_id):
    """
    指定された交差点の全流入レーンの待ち台数と待ち時間の合計を取得
    戻り値: (total_halting_number, total_waiting_time, state_vector)
    """
    state = []
    total_halt = 0
    total_wait = 0
    
    # 接続定義に基づいてレーン情報を取得
    if intersection_id not in INCOMING_EDGES:
        return 0, 0, np.zeros(24)

    for edge in INCOMING_EDGES[intersection_id]:
        for lane_idx in range(3): # 3レーン
            lane_id = f"{edge}_{lane_idx}"
            try:
                halt = traci.lane.getLastStepHaltingNumber(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                
                state.extend([halt, wait])
                total_halt += halt
                total_wait += halt * 1.0 # 報酬計算用の"待ち時間"定義に従う
            except traci.TraCIException:
                # レーンが存在しない場合のフォールバック
                state.extend([0, 0])
    
    return total_halt, total_wait, np.array(state, dtype=np.float32)

# ==========================================
# カスタムGym環境
# ==========================================
class SumoTrafficEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()
        
        # 行動空間: 各交差点4通りの行動 x 4交差点 = 256通りの組み合わせ
        self.action_space = spaces.Discrete(4 ** 4)
        
        # 状態空間: 72次元 * 4交差点 = 288次元
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(72 * 4,), dtype=np.float32
        )
        
        self.run_metrics = [] 
        self.current_step = 0
        self.prev_waiting_times = {i: 0.0 for i in INTERSECTIONS}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        try:
            traci.close()
        except:
            pass
        
        traci.start(SUMO_CMD)
        self.current_step = 0
        self.prev_waiting_times = {i: 0.0 for i in INTERSECTIONS}
        
        # 初期状態の取得
        return self._get_observation(), {}

    def _get_observation(self):
        raw_states = {}
        self.current_waiting_times = {} # 報酬計算用
        
        for tl in INTERSECTIONS:
            _, wait_val, vec = get_waiting_stats(tl)
            raw_states[tl] = vec
            self.current_waiting_times[tl] = wait_val
            
        full_obs = []
        for tl in INTERSECTIONS:
            # 自分の状態 (24)
            obs_part = list(raw_states[tl])
            # 隣接の状態 (24 * 2)
            for neighbor in NEIGHBORS[tl]:
                obs_part.extend(list(raw_states[neighbor]))
            full_obs.extend(obs_part)
            
        return np.array(full_obs, dtype=np.float32)

    def step(self, action):
        # 行動(0-255)を各交差点の行動(0-3)にデコード
        actions = {}
        temp_action = action
        for i, tl in enumerate(reversed(INTERSECTIONS)):
            actions[tl] = temp_action % 4
            temp_action //= 4
            
        # フェーズ時間の計算
        phase_durations = {}
        for tl in INTERSECTIONS:
            a = actions[tl]
            # [P0, P2, P5, P7] (修正後インデックス) の順に時間を決定
            durs = [MIN_GREEN] * 4
            durs[a] = 180 - (MIN_GREEN * 3)
            phase_durations[tl] = durs
            
        # SUMO制御ループ (1秒刻みで200ステップ)
        # 修正: フェーズインデックスを 0-9 の連番に修正
        # 0: NS青, 1: NS黄
        # 2: NS右青, 3: NS右黄, 4: 全赤
        # 5: EW青, 6: EW黄
        # 7: EW右青, 8: EW右黄, 9: 全赤
        
        for t in range(CYCLE_TIME):
            for tl in INTERSECTIONS:
                d = phase_durations[tl]
                t_cursor = 0
                
                # --- 南北 直進 ---
                # Phase 0: Green
                p_len = d[0]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 0)
                    continue
                t_cursor += p_len
                
                # Phase 1: Yellow (3s)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 1)
                    continue
                t_cursor += 3
                
                # --- 南北 右折 ---
                # Phase 2: Green (XMLのPhase 3に相当)
                p_len = d[1]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 2)
                    continue
                t_cursor += p_len
                
                # Phase 3: Yellow (3s) (XMLのPhase 4に相当)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 3)
                    continue
                t_cursor += 3
                
                # Phase 4: All Red (2s) (XMLのPhase 5に相当)
                if t < t_cursor + 2:
                    traci.trafficlight.setPhase(tl, 4)
                    continue
                t_cursor += 2

                # --- 東西 直進 ---
                # Phase 5: Green (XMLのPhase 6に相当)
                p_len = d[2]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 5)
                    continue
                t_cursor += p_len
                
                # Phase 6: Yellow (3s) (XMLのPhase 7に相当)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 6)
                    continue
                t_cursor += 3

                # --- 東西 右折 ---
                # Phase 7: Green (XMLのPhase 9に相当)
                p_len = d[3]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 7)
                    continue
                t_cursor += p_len

                # Phase 8: Yellow (3s) (XMLのPhase 10に相当)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 8)
                    continue
                t_cursor += 3

                # Phase 9: All Red (残り全て) (XMLのPhase 11に相当)
                traci.trafficlight.setPhase(tl, 9)

            traci.simulationStep()

        self.current_step += 1
        obs = self._get_observation()
        
        # 報酬計算
        total_reward = 0
        rewards_info = {}
        alpha = 0.5
        
        for tl in INTERSECTIONS:
            prev = self.prev_waiting_times[tl]
            curr = self.current_waiting_times[tl]
            diff_self = prev - curr
            
            neighbor_diffs = []
            for n in NEIGHBORS[tl]:
                n_prev = self.prev_waiting_times[n]
                n_curr = self.current_waiting_times[n]
                neighbor_diffs.append(n_curr - n_prev)
            
            avg_neighbor_worsening = np.mean(neighbor_diffs) if neighbor_diffs else 0
            
            numerator = diff_self - (alpha * avg_neighbor_worsening)
            
            if prev > 0:
                rate = numerator / prev
            else:
                rate = 0
                
            if rate >= 0.20:
                r = 2.0
            elif 0.05 <= rate < 0.20:
                r = 1.0
            elif -0.05 <= rate < 0.05:
                r = 0.0
            elif -0.20 <= rate < -0.05:
                r = -1.0
            else:
                r = -3.0
            
            total_reward += r
            rewards_info[tl] = r
            
        self.prev_waiting_times = self.current_waiting_times.copy()
        
        done = self.current_step >= STEPS_PER_EPISODE
        truncated = False
        
        if done:
            total_wait_sum = sum(self.current_waiting_times.values())
            self.run_metrics.append({
                "Episode": len(self.run_metrics) + 1,
                "C1": self.current_waiting_times["C1"],
                "C2": self.current_waiting_times["C2"],
                "C3": self.current_waiting_times["C3"],
                "C4": self.current_waiting_times["C4"],
                "Total": total_wait_sum
            })
            
        return obs, total_reward, done, truncated, {"rewards": rewards_info}
    
    def close(self):
        traci.close()

# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    env = SumoTrafficEnv()
    
    tmp_path = "./logs/"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["stdout", "csv"])

    # DQNモデルの設定
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        buffer_size=1000,
        batch_size=32, 
        gamma=0.99,
        verbose=1,
        # 探索率の設定
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.8, # 全ステップの80%を使って線形減衰させる
    )
    
    model.set_logger(new_logger)
    
    # EpsilonDecayCallbackは削除し、標準機能に任せる

    print(f"Starting training for {TOTAL_EPISODES} episodes...")
    # 毎エピソードログを表示するために log_interval=1 を維持
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    
    print("Training finished. Saving results to soro.csv...")
    df = pd.DataFrame(env.run_metrics)
    df.to_csv("soro.csv", index=False)
    print("Saved soro.csv")
    
    env.close()