import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Indy12 Robot Signal Analysis",
    layout="wide"
)

st.title("Indy12 Pick & Place Data Analysis Dashboard")

# =========================================
# 데이터 로드
# =========================================
@st.cache_data
def load_data():
    df_ideal = pd.read_csv("ideal_sim.csv")
    df_real = pd.read_csv("real_like.csv")
    df_clean = pd.read_csv("real_like_cleaned.csv")
    return df_ideal, df_real, df_clean

df_ideal, df_real, df_clean = load_data()

time_ideal = df_ideal["timestamp"].values
time_real = df_real["timestamp"].values
time_clean = df_clean["timestamp"].values

dt = float(np.median(np.diff(time_real)))

# =========================================
# 사이드바 옵션
# =========================================
st.sidebar.header("Options")

view_mode = st.sidebar.radio(
    "보기 모드 선택",
    ["단일 데이터 보기", "비교 보기 (ideal vs real vs cleaned)"]
)

joint_choice = st.sidebar.selectbox(
    "Joint 선택",
    options=["All"] + list(range(1, 7)),
    format_func=lambda x: "All joints" if x == "All" else f"Joint {x}"
)

# position → angle 로 명칭 변경
var_name = st.sidebar.selectbox(
    "변수 선택",
    options=["angle", "velocity", "torque"]
)

# 시간 범위 슬라이더 (real 기준)
t_min = float(time_real[0])
t_max = float(time_real[-1])

t_range = st.sidebar.slider(
    "시간 범위 [s]",
    min_value=t_min,
    max_value=t_max,
    value=(t_min, min(t_min + 15.0, t_max)),
    step=float(np.round((t_max - t_min) / 200.0, 3))
)

t_start, t_end = t_range

def select_range(time_arr, y_arr, t0, t1):
    mask = (time_arr >= t0) & (time_arr <= t1)
    return time_arr[mask], y_arr[mask]

# =========================================
# 단일 데이터 보기
# =========================================
if view_mode == "단일 데이터 보기":
    dataset_name = st.sidebar.selectbox(
        "Dataset 선택",
        options=["ideal_sim", "real_like (raw)", "real_like_cleaned"]
    )

    if dataset_name == "ideal_sim":
        df = df_ideal
        label_ds = "ideal_sim"
    elif dataset_name == "real_like (raw)":
        df = df_real
        label_ds = "real_like (raw)"
    else:
        df = df_clean
        label_ds = "real_like_cleaned"

    time = df["timestamp"].values

    # 변수별 컬럼 prefix 및 라벨
    if var_name == "angle":          # position → angle
        col_prefix = "joint_position_"
        y_label = "Joint angle [rad]"
    elif var_name == "velocity":
        col_prefix = "joint_velocity_"
        y_label = "Joint velocity [rad/s]"
    else:
        col_prefix = "joint_torque_"
        y_label = "Joint torque [arb. unit]"

    st.subheader(f"Dataset: {label_ds} | {var_name} | Joint {joint_choice}")

    fig, ax = plt.subplots(figsize=(12, 5))

    if joint_choice == "All":
        # Joint 1~6 모두 한 그래프에
        for j in range(1, 7):
            col = f"{col_prefix}{j}"
            y = df[col].values.astype(float)
            t_plot, y_plot = select_range(time, y, t_start, t_end)
            ax.plot(t_plot, y_plot, label=f"Joint {j}")
    else:
        # 단일 joint
        j = joint_choice
        col = f"{col_prefix}{j}"
        y = df[col].values.astype(float)
        t_plot, y_plot = select_range(time, y, t_start, t_end)
        ax.plot(t_plot, y_plot, label=f"Joint {j}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.markdown(
        f"- **데이터셋**: {label_ds}  \n"
        f"- **Joint 선택**: {joint_choice}  \n"
        f"- **Variable**: {var_name}"
    )

# =========================================
# 비교 보기: ideal vs real vs cleaned
# (여긴 여전히 Joint별 비교가 읽기 쉬워서 All 미지원)
# =========================================
else:
    if joint_choice == "All":
        st.warning("비교 보기에서는 Joint 하나씩 선택하는 게 더 명확해서, Joint 1~6 동시 플롯은 단일 데이터 보기에서만 지원합니다.")
        # 기본 Joint 1로 강제
        joint_idx = 1
    else:
        joint_idx = joint_choice

    # 변수별 컬럼 prefix 및 라벨
    if var_name == "angle":
        col = f"joint_position_{joint_idx}"
        y_label = "Joint angle [rad]"
    elif var_name == "velocity":
        col = f"joint_velocity_{joint_idx}"
        y_label = "Joint velocity [rad/s]"
    else:
        col = f"joint_torque_{joint_idx}"
        y_label = "Joint torque [arb. unit]"

    y_i = df_ideal[col].values.astype(float)
    y_r = df_real[col].values.astype(float)
    y_c = df_clean[col].values.astype(float)

    t_i, yi_plot = select_range(time_ideal, y_i, t_start, t_end)
    t_r, yr_plot = select_range(time_real, y_r, t_start, t_end)
    t_c, yc_plot = select_range(time_clean, y_c, t_start, t_end)

    st.subheader(f"비교 보기: Joint {joint_idx} | {var_name}")

    fig, ax = plt.subplots(figsize=(13, 4))

    # ideal: 파란 두꺼운 선
    ax.plot(t_i, yi_plot, label="ideal_sim", linewidth=2)

    # real: 빨간 실선
    ax.plot(t_r, yr_plot, color="red", linewidth=1.5,
            label="real_like (raw)")

    # cleaned: 주황 점선
    ax.plot(t_c, yc_plot, color="orange", linewidth=1,
            label="real_like_cleaned")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(y_label)
    ax.grid(True)

    # =============================
    # torque일 때 지연 추정 + 시각화
    # =============================
    if var_name == "torque":
        yi_full = df_ideal[col].values.astype(float)
        yr_full = df_clean[col].values.astype(float)  # cleaned 기준으로 delay 추정

        mask = (~np.isnan(yi_full)) & (~np.isnan(yr_full))
        if mask.sum() > 10:
            xi = yi_full[mask] - yi_full[mask].mean()
            xr = yr_full[mask] - yr_full[mask].mean()
            ti = time_ideal[mask]

            corr = np.correlate(xr, xi, mode="full")
            lags = np.arange(-len(xi) + 1, len(xi))
            lag_samples = lags[np.argmax(corr)]
            delay_est = lag_samples * dt

            peak_idx = np.argmax(np.abs(xi))
            t_peak_ideal = ti[peak_idx]
            t_peak_real = t_peak_ideal + delay_est

            ax.axvline(t_peak_ideal, color="blue", linestyle=":",
                       label="ideal peak")
            ax.axvline(t_peak_real, color="red", linestyle=":",
                       label=f"real peak (+{delay_est:.2f} s)")

            ax.text(
                0.02, 0.95,
                f"Estimated torque delay ≈ {delay_est:.3f} s",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            st.markdown(
                f"**토크 지연 추정 결과**: 약 `{delay_est:.3f} s` (크로스 코릴레이션 기반)"
            )

    ax.legend()
    st.pyplot(fig)

    st.markdown(
        "- 파란색: 이상적인 가상 공정 (`ideal_sim`)  \n"
        "- 빨간색: 현실 공정 로그 (`real_like`, raw)  \n"
        "- 주황 점선: 전처리 후 복원된 궤적 (`real_like_cleaned`)"
    )
