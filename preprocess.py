import numpy as np
import pandas as pd

# ============================================
#  0. 데이터 로드
# ============================================
df_real = pd.read_csv("real_like.csv")     # 현실 공정처럼 만든 데이터
df_ideal = pd.read_csv("ideal_sim.csv")    # 이상적인 가상공정 데이터

time = df_real["timestamp"].values
dt = float(np.round(np.mean(np.diff(time)), 4))
n = len(time)

# joint 관련 컬럼 이름 정리
pos_cols = [f"joint_position_{i}" for i in range(1, 7)]
vel_cols = [f"joint_velocity_{i}" for i in range(1, 7)]
tor_cols = [f"joint_torque_{i}" for i in range(1, 7)]

# ============================================
#  1. 변화율 기반 이상치(스파이크) 탐지 함수
#     - 인접 샘플 차분(Δx/Δt)을 보고,
#       통계적으로 너무 큰 지점은 스파이크로 판단
# ============================================

def mark_spikes_by_gradient(x, dt, k_sigma=5.0):
    """
    x: 1D 시계열 (NaN 포함 가능)
    dt: 샘플 간격
    k_sigma: 허용 표준편차 배수 (클수록 덜 민감)
    반환: spike 여부(bool) 배열
    """
    x = x.astype(float)
    mask_valid = ~np.isnan(x)
    idx = np.where(mask_valid)[0]

    spikes = np.zeros_like(x, dtype=bool)
    if len(idx) < 3:
        return spikes

    # 유효 구간에서 변화율 계산
    dx = np.diff(x[idx]) / dt
    # robust scale(MAD)로 스케일 추정
    med = np.median(dx)
    mad = np.median(np.abs(dx - med)) + 1e-9
    sigma = 1.4826 * mad  # 대략 표준편차

    thr = k_sigma * sigma

    large_grad = np.abs(dx - med) > thr
    spike_idx = idx[1:][large_grad]  # 오른쪽 인덱스를 스파이크로 본다

    spikes[spike_idx] = True
    # 스파이크 양 옆 한 칸씩도 같이 제거해서 더 자연스럽게 보간
    spikes[:-1] |= spikes[1:]
    spikes[1:] |= spikes[:-1]
    return spikes

# ============================================
#  2. 드롭(0 근처로 뚝 떨어지는 구간) 탐지 함수
#     - 속도/토크에서만 사용
# ============================================

def mark_drops(x, factor=0.1):
    """
    앞뒤에 비해 갑자기 매우 작아진(거의 0) 값들을 drop으로 판단.
    factor: 앞뒤 평균의 factor배 이하일 때 drop으로 간주.
    """
    x = x.astype(float)
    drops = np.zeros_like(x, dtype=bool)
    for i in range(1, len(x) - 1):
        if np.isnan(x[i-1]) or np.isnan(x[i]) or np.isnan(x[i+1]):
            continue
        neigh = 0.5 * (abs(x[i-1]) + abs(x[i+1])) + 1e-9
        if abs(x[i]) < factor * neigh:
            drops[i] = True
    return drops

# ============================================
#  3. 각 컬럼별로 스파이크/드롭/기존 NaN을 묶어서 "결측"으로 만들기
#     → 이후 한 번에 시간축 기반 보간
# ============================================

df_clean = df_real.copy()

for col_group, kind in [(pos_cols, "pos"), (vel_cols, "vel"), (tor_cols, "tor")]:
    for col in col_group:
        x = df_clean[col].values.astype(float)

        # 이미 NaN인 부분(센서 끊김)은 그대로 놓고
        base_nan = np.isnan(x)

        # (1) 변화율 기반 스파이크
        spikes = mark_spikes_by_gradient(x, dt, k_sigma=5.0)

        # (2) 드롭(속도/토크에만 적용)
        if kind in ("vel", "tor"):
            drops = mark_drops(x, factor=0.15)
        else:
            drops = np.zeros_like(spikes)

        # 이상치 마스크 종합
        bad = base_nan | spikes | drops

        # 이상치 구간을 NaN으로 만든다.
        x[bad] = np.nan
        df_clean[col] = x

# robot_state는 그대로 두되, 전처리 후에도 그대로 쓰고 싶으면 그대로 유지
# (원한다면 나중에 state도 변화율 기반으로 보정 가능)

# ============================================
#  4. 시간축 기준 선형 보간
#     - 각 컬럼별 NaN 구간을 앞뒤 값과 연결
#     - '실제 로봇이 물리적으로 갈 수 있는 궤적' 복원
# ============================================

df_interp = df_clean.copy()

for col in pos_cols + vel_cols + tor_cols:
    df_interp[col] = df_interp[col].interpolate(
        method="linear",
        limit_direction="both"
    )

# 결과 저장
df_interp.to_csv("real_like_cleaned.csv", index=False)
print("Saved real_like_cleaned.csv (스파이크/드롭/결측 보정 완료)")

# ============================================
#  5. torque 지연 0.2초 자동 추정 (cross-correlation)
#     - ideal_sim vs real_like_cleaned 비교
# ============================================

# 토크 한 joint씩 delay 추정해서 평균낸다.
delays = []

for j in range(1, 7):
    col_i = f"joint_torque_{j}"
    col_r = f"joint_torque_{j}"

    y_i = df_ideal[col_i].values.astype(float)
    y_r = df_interp[col_r].values.astype(float)

    # NaN 제거 공통 구간
    mask = (~np.isnan(y_i)) & (~np.isnan(y_r))
    if mask.sum() < 10:
        continue

    xi = y_i[mask] - y_i[mask].mean()
    xr = y_r[mask] - y_r[mask].mean()

    corr = np.correlate(xr, xi, mode="full")
    lags = np.arange(-len(xi) + 1, len(xi))
    lag_samples = lags[np.argmax(corr)]
    delay = lag_samples * dt
    delays.append(delay)
    print(f"joint {j}: estimated torque delay ≈ {delay:.4f} s")

if len(delays) > 0:
    delay_mean = float(np.mean(delays))
    print(f"\n[Summary] 평균 추정 토크 지연 ≈ {delay_mean:.4f} s")
else:
    print("delay를 추정할 수 있는 유효 토크 데이터가 충분하지 않습니다.")
