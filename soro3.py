import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import traci
import sumolib
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

# ==========================================
# 設定と定数
# ==========================================
SUMO_CMD = ["sumo", "-c", "4cross.sumocfg", "--no-warnings", "--no-step-log"]
# SUMO_CMD = ["sumo-gui", "-c", "4cross.sumocfg", "--no-warnings"] # GUIで見たい場合

EPISODE_LENGTH = 9000
CYCLE_TIME = 200
STEPS_PER_EPISODE = EPISODE_LENGTH // CYCLE_TIME
TOTAL_EPISODES = 10
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
# サイクル200秒 - 固定時間(黄3s*4 + 赤2s*4 = 20s) = 180秒 (利用可能)
# 実際にはフェーズ定義に合わせて調整:
# 黄色(3s)x4 = 12s, 全赤(2s)x2(XML定義上) = 4s -> 固定16s
# ユーザー要件: "固定の黄色信号3秒×4＋赤信号2秒×4を引いた180秒"
# ここではユーザー要件の「180秒」を優先し、4つの青フェーズに配分します。
# Action 0-3: 特定のフェーズを優先（最大化）し、他を最小(20s)にする
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
    state_vector: [lane0_halt, lane0_wait, lane1_halt, ...] (3 lanes * 4 edges * 2 metrics = 24 dims)
    """
    state = []
    total_halt = 0
    total_wait = 0
    
    for edge in INCOMING_EDGES[intersection_id]:
        for lane_idx in range(3): # 3レーン
            lane_id = f"{edge}_{lane_idx}"
            try:
                halt = traci.lane.getLastStepHaltingNumber(lane_id)
                # ユーザー要件: 待ち時間は HaltingNumber * 1 として扱う (報酬計算用)
                # ただし状態空間にはSUMOのWaitingTimeも入れるか、要件の定義に従う
                # "各レーンの待ち台数と待ち時間を取得する" -> SUMOのwaitingTimeも取得
                wait = traci.lane.getWaitingTime(lane_id)
                
                state.extend([halt, wait])
                total_halt += halt
                # 報酬計算用の"待ち時間"は halt * 1
                total_wait += halt * 1.0 
            except:
                state.extend([0, 0])
    
    return total_halt, total_wait, np.array(state, dtype=np.float32)

# ==========================================
# カスタムGym環境
# ==========================================
class SumoTrafficEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()
        
        # 4つの交差点を1つの「スーパーエージェント」として扱うか、
        # 中央集権的に制御するため、行動空間を結合します。
        # 各交差点4通りの行動 x 4交差点 = 256通りの組み合わせ
        self.action_space = spaces.Discrete(4 ** 4)
        
        # 状態空間: 4交差点分 × (自分72次元) 
        # ※SB3は標準で単一エージェント用のため、全交差点の情報を結合して入力とします。
        # 各交差点の状態: 自分(24次元) + 隣接2つ(24次元x2) = 72次元
        # 全体: 72 * 4 = 288次元
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(72 * 4,), dtype=np.float32
        )
        
        self.run_metrics = [] # エピソードごとの結果保存用
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
        
        return self._get_observation(), {}

    def _get_observation(self):
        # まず全交差点の生データを取得
        raw_states = {}
        self.current_waiting_times = {} # 報酬計算用
        
        for tl in INTERSECTIONS:
            _, wait_val, vec = get_waiting_stats(tl)
            raw_states[tl] = vec
            self.current_waiting_times[tl] = wait_val
            
        # 各交差点ごとの72次元ベクトルを作成し結合
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
        # 基数4で分解
        actions = {}
        temp_action = action
        for i, tl in enumerate(reversed(INTERSECTIONS)): # C4, C3, C2, C1
            actions[tl] = temp_action % 4
            temp_action //= 4
            
        # シミュレーション実行 (200秒分)
        # フェーズ構成: 
        # P0(NS直):Act, P1(Y):3, P3(NS右):Act, P4(Y):3, P5(R):2, 
        # P6(EW直):Act, P7(Y):3, P9(EW右):Act, P10(Y):3, P11(R):2
        # 固定時間合計: 3+3+2+3+3+2 = 16秒 (XML定義に準拠)
        # 残り 184秒 を配分 (ユーザー要件の180秒に近づけるため、余剰は補正に使用)
        
        # 配分ロジック:
        # 行動0: NS直進優先, 行動1: NS右折優先, 行動2: EW直進優先, 行動3: EW右折優先
        # 優先フェーズには (180 - 20*3) = 120秒, 他は20秒 を割り当て
        
        phase_durations = {}
        for tl in INTERSECTIONS:
            a = actions[tl]
            # [P0, P3, P6, P9] の順に時間を決定
            durs = [MIN_GREEN] * 4
            durs[a] = 180 - (MIN_GREEN * 3) # 残りを全て優先フェーズへ
            phase_durations[tl] = durs
            
        # SUMO制御ループ (1秒刻みで200ステップ)
        # 各フェーズの切り替えタイミングを計算して実行
        # フェーズ順序: 0(G)->1(y)->3(G)->4(y)->5(r)->6(G)->7(y)->9(G)->10(y)->11(r)
        
        # フェーズ定義リスト (index, duration)
        # Variable phases are indices 0, 3, 6, 9
        
        current_phases = {tl: 0 for tl in INTERSECTIONS} # 現在のフェーズインデックス
        cycle_plan = []
        
        # タイムラインの構築
        for t in range(CYCLE_TIME):
            # 各交差点について、現在の時刻tがどのフェーズにあるか判定してsetPhase
            for tl in INTERSECTIONS:
                d = phase_durations[tl]
                # 累積時間で判定
                t_cursor = 0
                
                # Phase 0 (NS G)
                p_len = d[0]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 0)
                    continue
                t_cursor += p_len
                
                # Phase 1 (Y)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 1)
                    continue
                t_cursor += 3
                
                # Phase 3 (NS R_G)
                p_len = d[1]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 3)
                    continue
                t_cursor += p_len
                
                # Phase 4 (Y)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 4)
                    continue
                t_cursor += 3
                
                # Phase 5 (R)
                if t < t_cursor + 2:
                    traci.trafficlight.setPhase(tl, 5)
                    continue
                t_cursor += 2

                # Phase 6 (EW G)
                p_len = d[2]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 6)
                    continue
                t_cursor += p_len
                
                # Phase 7 (Y)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 7)
                    continue
                t_cursor += 3

                # Phase 9 (EW R_G)
                p_len = d[3]
                if t < t_cursor + p_len:
                    traci.trafficlight.setPhase(tl, 9)
                    continue
                t_cursor += p_len

                # Phase 10 (Y)
                if t < t_cursor + 3:
                    traci.trafficlight.setPhase(tl, 10)
                    continue
                t_cursor += 3

                # Phase 11 (R)
                # 残り時間すべて
                traci.trafficlight.setPhase(tl, 11)

            traci.simulationStep()

        # ステップ終了後の観測
        self.current_step += 1
        obs = self._get_observation()
        
        # 報酬計算
        total_reward = 0
        rewards_info = {}
        alpha = 0.5
        
        for tl in INTERSECTIONS:
            # 自分
            prev = self.prev_waiting_times[tl]
            curr = self.current_waiting_times[tl]
            diff_self = prev - curr # 正なら減少(良)
            
            # 隣接
            neighbor_diffs = []
            for n in NEIGHBORS[tl]:
                n_prev = self.prev_waiting_times[n]
                n_curr = self.current_waiting_times[n]
                neighbor_diffs.append(n_curr - n_prev) # 増加量(正なら悪化)
            
            avg_neighbor_worsening = np.mean(neighbor_diffs) if neighbor_diffs else 0
            
            # 減少率の計算
            # 式: ( (前回 - 今回) - α(隣接変化平均) ) / 前回
            numerator = diff_self - (alpha * avg_neighbor_worsening)
            
            if prev > 0:
                rate = numerator / prev
            else:
                rate = 0 # 前回0なら変化なし扱い
                
            # 報酬テーブル
            if rate >= 0.20:
                r = 2.0
            elif 0.05 <= rate < 0.20:
                r = 1.0
            elif -0.05 <= rate < 0.05:
                r = 0.0
            elif -0.20 <= rate < -0.05:
                r = -1.0
            else: # rate < -0.20
                r = -3.0
            
            total_reward += r
            rewards_info[tl] = r
            
        # 更新
        self.prev_waiting_times = self.current_waiting_times.copy()
        
        done = self.current_step >= STEPS_PER_EPISODE
        truncated = False
        
        # ログ用データ収集
        if done:
            total_wait_sum = sum(self.current_waiting_times.values())
            self.run_metrics.append({
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
# イプシロン減衰コールバック
# ==========================================
class EpsilonDecayCallback(BaseCallback):
    def __init__(self, total_timesteps, initial_eps=1.0, final_eps=0.05, decay_fraction=0.8, verbose=0):
        super(EpsilonDecayCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.decay_fraction = decay_fraction
        self.decay_steps = int(total_timesteps * decay_fraction)

    def _on_step(self) -> bool:
        # 現在の進捗割合
        current_step = self.num_timesteps
        if current_step < self.decay_steps:
            progress = current_step / self.decay_steps
            new_eps = self.initial_eps - progress * (self.initial_eps - self.final_eps)
        else:
            new_eps = self.final_eps
            
        self.model.exploration_rate = new_eps
        return True

# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    # 環境の初期化
    env = SumoTrafficEnv()
    
    # ロガーの設定
    tmp_path = "./logs/"
    os.makedirs(tmp_path, exist_ok=True)
    new_logger = configure(tmp_path, ["stdout", "csv"])

    # モデルの定義 (DQN)
    # 探索率はコールバックで制御するため、初期値は適当に設定して上書きします
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=1e-3, 
        buffer_size=1000,
        batch_size=32, 
        gamma=0.99,
        verbose=1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        # SB3の標準減衰を使わず、カスタムコールバックで制御します
    )
    
    model.set_logger(new_logger)
    
    # コールバックの準備 (探索率制御)
    # model.learnのtotal_timestepsに対して80%で減衰
    epsilon_callback = EpsilonDecayCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        initial_eps=1.0,
        final_eps=0.05,
        decay_fraction=0.8
    )

    print(f"Starting training for {TOTAL_EPISODES} episodes...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=epsilon_callback)
    
    # 完了後のCSV保存
    print("Training finished. Saving results to soro.csv...")
    df = pd.DataFrame(env.run_metrics)
    df.to_csv("soro.csv", index=False)
    print("Saved soro.csv")
    
    env.close()