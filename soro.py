import os
import sys
import time
import random
import itertools
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# SUMO_HOME check
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
# --- 追加: Loggerの設定用 ---
from stable_baselines3.common.logger import configure

# --- Configuration ---
# 警告メッセージを抑制するために "--no-warnings" を追加
SUMO_CMD = ["sumo", "-c", "4cross.sumocfg", "--no-step-log", "true", "--waiting-time-memory", "10000", "--no-warnings"]
# GUIで確認したい場合は "sumo" を "sumo-gui" に変更してください
# SUMO_CMD = ["sumo-gui", "-c", "4cross.sumocfg", "--no-step-log", "true", "--waiting-time-memory", "10000", "--no-warnings"]

EPISODES = 10
SIM_DURATION = 9000
CYCLE_TIME = 200
CONTROL_STEP = 200 # 行動決定間隔

# フェーズ時間の定義
YELLOW_TIME = 4
RED_TIME = 2
MIN_GREEN = 20
# 固定時間合計 = (黄4s * 4) + (赤2s * 2) = 20秒
# 可変青時間合計 = 200 - 20 = 180秒
TOTAL_GREEN_BUDGET = 180 
GREEN_STEP = 20

# 交差点定義
JUNCTIONS = ["C1", "C2", "C3", "C4"]

# フェーズ状態定義 (XMLの記述に基づく)
# 順番: 
# 0: NS直(G) -> 1: NS直(Y) -> 
# 2: NS右(G) -> 3: NS右(Y) -> 4: 全赤 -> 
# 5: EW直(G) -> 6: EW直(Y) -> 
# 7: EW右(G) -> 8: EW右(Y) -> 9: 全赤
PHASE_STATES = [
    "GGgrrrGGgrrr", # 0: NS Straight Green (Variable)
    "yyyrrryyyrrr", # 1: NS Straight Yellow (Fixed 4s)
    "rrGrrrrrGrrr", # 2: NS Right Green (Variable)
    "rryrrrrryrrr", # 3: NS Right Yellow (Fixed 4s)
    "rrrrrrrrrrrr", # 4: All Red (Fixed 2s)
    "rrrGGgrrrGGg", # 5: EW Straight Green (Variable)
    "rrryyyrrryyy", # 6: EW Straight Yellow (Fixed 4s)
    "rrrrrGrrrrrG", # 7: EW Right Green (Variable)
    "rrrrryrrrrry", # 8: EW Right Yellow (Fixed 4s)
    "rrrrrrrrrrrr"  # 9: All Red (Fixed 2s)
]

# 各交差点の流入レーン定義 (Observation取得用 3レーンx4方向)
LANES = {
    "C1": ["N1_C1_0", "N1_C1_1", "N1_C1_2", "W1_C1_0", "W1_C1_1", "W1_C1_2", "C2_C1_0", "C2_C1_1", "C2_C1_2", "C3_C1_0", "C3_C1_1", "C3_C1_2"],
    "C2": ["N2_C2_0", "N2_C2_1", "N2_C2_2", "E1_C2_0", "E1_C2_1", "E1_C2_2", "C1_C2_0", "C1_C2_1", "C1_C2_2", "C4_C2_0", "C4_C2_1", "C4_C2_2"],
    "C3": ["S1_C3_0", "S1_C3_1", "S1_C3_2", "W2_C3_0", "W2_C3_1", "W2_C3_2", "C1_C3_0", "C1_C3_1", "C1_C3_2", "C4_C3_0", "C4_C3_1", "C4_C3_2"],
    "C4": ["S2_C4_0", "S2_C4_1", "S2_C4_2", "E2_C4_0", "E2_C4_1", "E2_C4_2", "C2_C4_0", "C2_C4_1", "C2_C4_2", "C3_C4_0", "C3_C4_1", "C3_C4_2"]
}

# --- Action Space Generation ---
def generate_action_space():
    # 180秒を4つのフェーズ(NS直, NS右, EW直, EW右)に分配。各最小20秒。
    # x1 + x2 + x3 + x4 = 5 (unit: 20s, base=20s) -> Total 180s
    combinations = []
    # 0~5の配分を4箇所に割り振る (合計5になる組み合わせ)
    for p in itertools.product(range(6), repeat=4):
        if sum(p) == 5:
            # 各フェーズの長さ = 最小20秒 + (配分単位 * 20秒)
            durations = [MIN_GREEN + x * GREEN_STEP for x in p]
            combinations.append(durations)
    return combinations

ACTION_MAP = generate_action_space()
NUM_ACTIONS = len(ACTION_MAP)
print(f"Action Space Size: {NUM_ACTIONS} (Expected 56)")

# --- Dummy Gym Env for SB3 Initialization ---
# SB3のモデル初期化時にのみ使用するダミー環境
# これにより、複雑なSUMO環境を直接渡さずにDQNモデルを作成できます
class DummyGymEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # 状態空間: 24次元 (waiting_count, waiting_time normalized)
        self.observation_space = spaces.Box(low=0, high=1, shape=(24,), dtype=np.float32)
        # 行動空間: 56通り
        self.action_space = spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        return np.zeros(24, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(24, dtype=np.float32), 0.0, False, False, {}

# --- Real SUMO Environment Manager ---
# 実際のシミュレーションとデータのやり取りを行うクラス
class Sumo4CrossEnv:
    def __init__(self):
        self.junctions = JUNCTIONS
        self.lanes = LANES
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        
        self.prev_cost = {j: 0.0 for j in self.junctions}
        self.total_waiting_time = {j: 0.0 for j in self.junctions} # エピソードごとの集計用
        self.steps = 0
        self.sim_time = 0
        self._current_cost_cache = {}
        
    def start(self):
        traci.start(SUMO_CMD)
        
    def close(self):
        traci.close()

    def reset(self):
        self.prev_cost = {j: 0.0 for j in self.junctions}
        self.total_waiting_time = {j: 0.0 for j in self.junctions}
        self.steps = 0
        self.sim_time = 0
        self._current_cost_cache = {j: 0.0 for j in self.junctions}
        return self._get_observations()

    def _get_observations(self):
        obs_dict = {}
        for j in self.junctions:
            obs = []
            current_cost = 0.0
            for lane in self.lanes[j]:
                try:
                    halt_num = traci.lane.getLastStepHaltingNumber(lane)
                    wait_time = traci.lane.getWaitingInformation(lane)["waitingTime"]
                    
                    # 正規化 (仮定値: 最大台数20, 最大待ち時間200sでスケーリング)
                    obs.append(min(1.0, halt_num / 20.0)) 
                    obs.append(min(1.0, wait_time / 200.0)) 
                    
                    # コスト計算: 要件「getLastStepHaltingNumber×1」
                    current_cost += halt_num * 1.0
                except:
                    obs.append(0.0)
                    obs.append(0.0)
            
            obs_dict[j] = np.array(obs, dtype=np.float32)
            self._current_cost_cache[j] = current_cost
            
        return obs_dict

    def step(self, actions):
        """
        actions: { "C1": action_idx, "C2": action_idx, ... }
        """
        # 1. 信号プログラムの作成と適用
        for j in self.junctions:
            act_idx = actions[j]
            durations = ACTION_MAP[act_idx] # [ns_s, ns_r, ew_s, ew_r]
            
            phases = []
            # NS Straight
            phases.append(traci.trafficlight.Phase(durations[0], PHASE_STATES[0])) # Green
            phases.append(traci.trafficlight.Phase(YELLOW_TIME,  PHASE_STATES[1])) # Yellow
            # NS Right
            phases.append(traci.trafficlight.Phase(durations[1], PHASE_STATES[2])) # Green
            phases.append(traci.trafficlight.Phase(YELLOW_TIME,  PHASE_STATES[3])) # Yellow
            # All Red
            phases.append(traci.trafficlight.Phase(RED_TIME,     PHASE_STATES[4])) # Red
            
            # EW Straight
            phases.append(traci.trafficlight.Phase(durations[2], PHASE_STATES[5])) # Green
            phases.append(traci.trafficlight.Phase(YELLOW_TIME,  PHASE_STATES[6])) # Yellow
            # EW Right
            phases.append(traci.trafficlight.Phase(durations[3], PHASE_STATES[7])) # Green
            phases.append(traci.trafficlight.Phase(YELLOW_TIME,  PHASE_STATES[8])) # Yellow
            # All Red
            phases.append(traci.trafficlight.Phase(RED_TIME,     PHASE_STATES[9])) # Red
            
            logic = traci.trafficlight.Logic(f"program_{self.steps}", 0, 0, phases)
            traci.trafficlight.setCompleteRedYellowGreenDefinition(j, logic)

        # 2. シミュレーション実行 (200秒分)
        # ステップごとに待ち時間を集計する
        for _ in range(CONTROL_STEP):
            traci.simulationStep()
            self.sim_time += 1
            
            # 統計収集: 各交差点の総待ち時間（全レーンのHaltingNumberの和）
            for j in self.junctions:
                cost = 0
                for lane in self.lanes[j]:
                   cost += traci.lane.getLastStepHaltingNumber(lane)
                self.total_waiting_time[j] += cost

        # 3. 報酬計算と次状態取得
        next_obs_dict = self._get_observations()
        rewards = {}
        
        for j in self.junctions:
            prev = self.prev_cost[j]
            current = self._current_cost_cache[j]
            
            # ReductionRate = (前回 - 今回) / 前回
            if prev == 0:
                rate = 0.0 if current == 0 else -1.0 # 前回0で今回増えたら悪化
            else:
                rate = (prev - current) / prev
            
            # 報酬テーブル
            if rate >= 0.2:
                r = 2.0
            elif 0.05 <= rate < 0.2:
                r = 1.0
            elif -0.05 <= rate < 0.05:
                r = 0.0
            elif -0.2 < rate < -0.05:
                r = -1.0
            else: # rate <= -0.2 or other worsening
                r = -3.0
            
            rewards[j] = r
            self.prev_cost[j] = current

        self.steps += 1
        done = self.sim_time >= SIM_DURATION
        
        return next_obs_dict, rewards, done, {}

# --- Epsilon Helper ---
def get_epsilon(total_steps_fraction, start_eps=1.0, end_eps=0.05, decay_fraction=0.8):
    if total_steps_fraction >= decay_fraction:
        return end_eps
    return start_eps - (start_eps - end_eps) * (total_steps_fraction / decay_fraction)

# --- Main Training Loop ---
if __name__ == "__main__":
    # 1. DQNモデルの準備 (DummyEnvを使用)
    dummy_env = DummyGymEnv() 
    
    # 完全独立エージェント: 交差点ごとに個別のDQNモデルを作成
    agents = {}
    print("Initializing Agents...")
    for j in JUNCTIONS:
        agents[j] = DQN("MlpPolicy", dummy_env, verbose=0, learning_rate=1e-3, buffer_size=50000, batch_size=64)
        # --- 重要: 手動学習ループのためにLoggerを設定 ---
        # format_strings=[] を渡すことでログ出力を抑制(エラー回避)
        agents[j].set_logger(configure(format_strings=[]))
    
    # 2. 実際のシミュレーション環境
    env = Sumo4CrossEnv()
    
    # 統計用データフレーム準備
    results_data = []

    print(f"Starting Training for {EPISODES} episodes...")
    
    # 全学習ステップ数 (Epsilon減衰計算用)
    total_timesteps = EPISODES * (SIM_DURATION // CONTROL_STEP)
    current_global_step = 0

    for episode in range(EPISODES):
        print(f"--- Episode {episode + 1}/{EPISODES} ---")
        
        env.start()
        obs = env.reset()
        done = False
        
        while not done:
            actions = {}
            # Epsilon-Greedy
            epsilon = get_epsilon(current_global_step / total_timesteps)
            
            # 各エージェントの行動決定
            for j in JUNCTIONS:
                if random.random() < epsilon:
                    actions[j] = env.action_space.sample()
                else:
                    action, _ = agents[j].predict(obs[j], deterministic=True)
                    actions[j] = action.item()
            
            # 環境ステップ実行
            next_obs, rewards, done, _ = env.step(actions)
            
            # 各エージェントの学習
            for j in JUNCTIONS:
                # バッチ次元(1, N)を追加してReplayBufferに保存
                _obs = obs[j].reshape(1, -1)
                _next_obs = next_obs[j].reshape(1, -1)
                _action = np.array([actions[j]])
                _reward = np.array([rewards[j]])
                _done = np.array([done])
                
                agents[j].replay_buffer.add(
                    _obs, _next_obs, _action, _reward, _done, [{}]
                )
                
                # バッファがある程度溜まったら学習
                if agents[j].replay_buffer.size() > 100:
                    agents[j].train(gradient_steps=1, batch_size=64)
            
            obs = next_obs
            current_global_step += 1
        
        env.close()
        
        # エピソード結果の集計
        episode_stats = {"Episode": episode + 1}
        for j in JUNCTIONS:
            # 平均待ち時間 = 総HaltingNumber積算値 / シミュレーション秒数 (またはステップ数)
            # ここではシンプルにシミュレーション時間あたりの平均待機台数指標として記録
            avg_wait = env.total_waiting_time[j] / SIM_DURATION
            episode_stats[j] = avg_wait
            print(f"  {j} Total Waiting Cost: {env.total_waiting_time[j]:.1f}, Avg Score: {avg_wait:.2f}")
        
        results_data.append(episode_stats)
        print(f"  Epsilon: {epsilon:.4f}")

    # 3. 結果の保存 (CSVに変更)
    df = pd.DataFrame(results_data)
    # カラム順序を整理
    cols = ["Episode"] + JUNCTIONS
    df = df[cols]
    
    # openpyxlがない環境に対応するため、CSVで保存します
    output_file = "soro.csv"
    df.to_csv(output_file, index=False)
    print(f"\nTraining finished. Results saved to '{output_file}'.")
    print(df)