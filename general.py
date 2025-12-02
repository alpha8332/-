import numpy as np
import pandas as pd

# ======================================
#  1) 기존 pick&place joint angle 불러오기
# ======================================
src = "indy12_pickplace_joint_angles.csv"

df_angle = pd.read_csv(src)
time = df_angle["time"].values
dt = float(np.round(np.mean(np.diff(time)), 4))
n = len(time)

n_joints = 6
pos_cols = [f"joint angle{i}" for i in range(n_joints)]
joint_pos = df_angle[pos_cols].values  # [n, 6]

# ======================================
#  2) ideal_sim: 속도/토크 계산 (이상적인 모델)
# ======================================

# (1) 속도: 중앙차분 (양 끝은 1차)
joint_vel = np.zeros_like(joint_pos)
joint_vel[1:-1, :] = (joint_pos[2:, :] - joint_pos[:-2, :]) / (2 * dt)
joint_vel[0, :] = (joint_pos[1, :] - joint_pos[0, :]) / dt
joint_vel[-1, :] = (joint_pos[-1, :] - joint_pos[-2, :]) / dt

# (2) 토크: 간단한 스프링+댐퍼 모델
#     τ = K (q - q0) + C q_dot
#     q0는 전체 평균 각도(정지 자세)로 둠
q0 = np.mean(joint_pos, axis=0)

K = np.array([40, 35, 30, 25, 20, 15], dtype=float)  # spring 계수
C = np.array([3.0, 3.0, 2.5, 2.5, 2.0, 2.0], dtype=float)  # damping 계수

joint_torque = np.zeros_like(joint_pos)
for j in range(n_joints):
    joint_torque[:, j] = (
        K[j] * (joint_pos[:, j] - q0[j]) +
        C[j] * joint_vel[:, j]
    )

# (3) robot_state: 사이클 구조 그대로 running/idle로 구분
#     여기서는 단순히: 움직이는 구간 RUN(1), 거의 안 움직이면 IDLE(2)
speed_norm = np.linalg.norm(joint_vel, axis=1)
th = np.percentile(speed_norm, 40)  # 하위 40%를 idle로
robot_state_ideal = np.where(speed_norm > th, 1, 2).astype(int)

# (4) ideal_sim DataFrame 구성
data_ideal = {"timestamp": time, "robot_state": robot_state_ideal}
for j in range(n_joints):
    data_ideal[f"joint_position_{j+1}"] = joint_pos[:, j]
for j in range(n_joints):
    data_ideal[f"joint_velocity_{j+1}"] = joint_vel[:, j]
for j in range(n_joints):
    data_ideal[f"joint_torque_{j+1}"] = joint_torque[:, j]

df_ideal = pd.DataFrame(data_ideal)
df_ideal.to_csv("ideal_sim.csv", index=False)
print("Saved ideal_sim.csv")

# ======================================
#  3) real_like: 현실 공정 느낌의 노이즈/결함 추가
# ======================================

rng = np.random.default_rng(42)

pos_real = joint_pos.copy()
vel_real = joint_vel.copy()
tor_real = joint_torque.copy()
state_real = robot_state_ideal.copy()

# (A) 위치: 백래시/마찰 → cycle 별 offset + 작은 노이즈
T_total = time[-1] - time[0]
cycle_T = 6.0    # 한 pick&place cycle이 6초라고 가정 (대략)
n_cycles = int(np.floor(T_total / cycle_T)) + 1

for j in range(n_joints):
    # 전체 구간에 작은 가우시안 노이즈 (±0.2deg 수준)
    noise = rng.normal(scale=np.deg2rad(0.2), size=n)
    pos_real[:, j] += noise

    # cycle별 offset (백래시 느낌)
    for cyc in range(n_cycles):
        ts = time[0] + cyc * cycle_T
        te = ts + cycle_T
        idx_cyc = (time >= ts) & (time < te)
        if not np.any(idx_cyc):
            continue
        offset = rng.normal(scale=np.deg2rad(0.3))
        pos_real[idx_cyc, j] += offset

# (B) 속도: 센서 노이즈 & 드롭
for j in range(n_joints):
    # 속도 노이즈 (±5% 정도)
    vel_real[:, j] *= (1.0 + rng.normal(scale=0.05, size=n))
    vel_real[:, j] += rng.normal(scale=0.02, size=n)

# 드롭: 몇 지점은 속도가 0처럼 찍히게
for _ in range(12):
    i = rng.integers(0, n)
    j = rng.integers(0, n_joints)
    vel_real[i, j] = 0.0

# (C) 토크: 마찰/백래시 + 스파이크 + 저주파 노이즈 + 0.2초 지연
tor_real = np.zeros_like(joint_torque)

max_K = max(K)
base_noise = 0.5  # joint1 기준 노이즈 크기

for j in range(n_joints):
    # 1) 기본 토크 + 마찰계수 오차
    bias = 1.0 + rng.normal(scale=0.05)   # 너무 크지 않게 5% 정도
    tau = bias * joint_torque[:, j]

    # 2) 저주파 노이즈 (센서/기구 backlash 느낌)
    #   - high freq 화이트 노이즈 대신, exp kernel로 low-pass
    white = rng.normal(scale=1.0, size=n)
    alpha = 0.02  # 필터 계수(작을수록 저주파)
    lp = np.zeros_like(white)
    for k in range(1, n):
        lp[k] = (1 - alpha) * lp[k-1] + alpha * white[k]

    # 노이즈 크기는 토크 스케일에 비례 (joint6에서 너무 커지지 않도록)
    noise_scale = base_noise * (K[j] / max_K)   # 큰 관절 > 작은 관절
    tau += noise_scale * lp

    tor_real[:, j] = tau

# 3) 스파이크: 랜덤 지점 몇 개에서 2~3배 튀게
for _ in range(10):
    i = rng.integers(0, n)
    j = rng.integers(0, n_joints)
    tor_real[i, j] *= rng.uniform(2.0, 3.0)

# 4) 0.2초 지연 적용
delay_sec = 0.2
delay_steps = int(round(delay_sec / dt))
tor_delayed = np.full_like(tor_real, np.nan)
tor_delayed[delay_steps:, :] = tor_real[:-delay_steps, :]

# (D) 결측 구간 (센서 통신 끊김)
nan_segments = [(8.0, 8.4), (18.5, 18.9), (32.0, 32.5)]
for (ts, te) in nan_segments:
    idx = (time >= ts) & (time <= te)
    # 3번, 4번 관절을 일부러 더 망가뜨리기
    pos_real[idx, 2:4] = np.nan
    vel_real[idx, 2:4] = np.nan
    tor_delayed[idx, 2:4] = np.nan

# (E) Robot state: 일부 구간 stop / idle 강제 삽입
for ts, te, st in [(5.0, 6.0, 2), (15.0, 16.0, 0), (25.0, 27.0, 2)]:
    idx = (time >= ts) & (time <= te)
    state_real[idx] = st

# (F) DataFrame 구성
data_real = {"timestamp": time, "robot_state": state_real}
for j in range(n_joints):
    data_real[f"joint_position_{j+1}"] = pos_real[:, j]
for j in range(n_joints):
    data_real[f"joint_velocity_{j+1}"] = vel_real[:, j]
for j in range(n_joints):
    data_real[f"joint_torque_{j+1}"] = tor_delayed[:, j]

df_real = pd.DataFrame(data_real)
df_real.to_csv("real_like.csv", index=False)
print("Saved real_like.csv")
