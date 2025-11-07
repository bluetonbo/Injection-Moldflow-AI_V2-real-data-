import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 기본 설정 ---
st.set_page_config(page_title="Injection Moldflow AI", layout="wide")

# --- 모델 로드 ---
MODEL_PATH = "model/xgb_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 모델 로드 실패: {e}")
    st.stop()

# --- 기본값 초기화 ---
default_vals = {
    'T_Melt': 240.0,
    'V_Inj': 5.0,
    'P_Pack': 80.0,
    'T_Mold': 50.0,
    'Meter': 30.0,
    'VP_Switch_Pos': 10.0,
}

# --- 세션 상태 초기화 ---
for key, val in default_vals.items():
    if key not in st.session_state:
        st.session_state[f'input_{key}'] = val

# --- 제목 ---
st.title("💡 사출 성형 AI 예측 시스템 (Injection Moldflow AI)")

# --- 입력값 슬라이더 영역 ---
st.subheader("🔧 공정 변수 입력")

col_melt, col_inj, col_pack = st.columns(3)
col_mold, col_meter, col_vp = st.columns(3)

input_vars = {}

with col_melt:
    input_vars['T_Melt'] = st.slider(
        '용융 온도 (T_Melt)',
        min_value=230.0,
        max_value=260.0,
        value=float(st.session_state['input_T_Melt']),
        step=5.0,
        key='slider_T_Melt'
    )

with col_inj:
    input_vars['V_Inj'] = st.slider(
        '사출 속도 (V_Inj)',
        min_value=1.0,
        max_value=10.0,
        value=float(st.session_state['input_V_Inj']),
        step=1.0,
        key='slider_V_Inj'
    )

with col_pack:
    input_vars['P_Pack'] = st.slider(
        '보압 (P_Pack)',
        min_value=50.0,
        max_value=100.0,
        value=float(st.session_state['input_P_Pack']),
        step=5.0,
        key='slider_P_Pack'
    )

with col_mold:
    input_vars['T_Mold'] = st.slider(
        '금형 온도 (T_Mold)',
        min_value=30.0,
        max_value=80.0,
        value=float(st.session_state['input_T_Mold']),
        step=5.0,
        key='slider_T_Mold'
    )

with col_meter:
    input_vars['Meter'] = st.slider(
        '계량 위치 (Meter)',
        min_value=10.0,
        max_value=50.0,
        value=float(st.session_state['input_Meter']),
        step=5.0,
        key='slider_Meter'
    )

with col_vp:
    input_vars['VP_Switch_Pos'] = st.slider(
        '전환 위치 (V/P Switch Pos)',
        min_value=5.0,
        max_value=20.0,
        value=float(st.session_state['input_VP_Switch_Pos']),
        step=1.0,
        key='slider_VP_Switch_Pos'
    )

# --- 예측 버튼 ---
st.markdown("---")
if st.button("🔮 예측 실행"):
    X_input = pd.DataFrame([input_vars])
    try:
        y_pred = model.predict(X_input)
        st.success(f"✅ 예측 결과: {y_pred[0]:.3f}")
    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")

# --- 디버그용 데이터 확인 ---
with st.expander("입력 변수 보기"):
    st.dataframe(pd.DataFrame([input_vars]))

