import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize 

# =================================================================
# 0. A.py의 설정 및 상수 (B.py의 GUI 환경에 맞게 조정)
# =================================================================

st.set_page_config(layout="wide", page_title="Weld Line AI 진단 시스템 (A+B 통합)")

# A.py의 프로세스 변수 정의
PROCESS_VARS = ['T_Melt', 'V_Inj', 'P_Pack', 'T_Mold', 'Meter', 'VP_Switch_Pos']
TARGET_VAR = 'Y_Weld'

# A.py의 기본값 정의 (사용자 요청에 따라 유지)
A_DEFAULT_INPUT_VALS = {
    'T_Melt': 230.0, 'V_Inj': 3.0, 'P_Pack': 70.0, 
    'T_Mold': 50.0, 'Meter': 195.0, 'VP_Switch_Pos': 14.0
}

# A.py의 기본값을 수용할 수 있도록 슬라이더 범위 조정
A_VARIABLE_BOUNDS = {
    # (min, max, step)
    'T_Melt': (200, 300, 1), 
    'V_Inj': (0, 150, 1),
    'P_Pack': (50, 120, 1),
    'T_Mold': (30, 120, 1),
    'Meter': (100.0, 300.0, 0.1), 
    'VP_Switch_Pos': (5.0, 20.0, 0.1) 
}

# 최종 사용할 피처 목록 (A.py 모델 구조: 6개 프로세스 변수 + 2개 파생 변수)
FEATURES = PROCESS_VARS + ['T_Weld', 't_Fill']

# =================================================================
# 1. 데이터 로딩 및 모델 학습 로직 (A.py 기반으로 단순화)
# =================================================================

def load_df_from_uploader(uploaded_file):
    """업로드된 파일 객체에서 Pandas DataFrame을 로드합니다."""
    if uploaded_file is not None:
        try:
            return pd.read_excel(uploaded_file, engine='openpyxl')
        except Exception as e:
            st.error(f"⚠️ 파일 로드 오류: {e}")
            return None
    return None

def process_weld_data(df_virtual, df_real):
    """업로드된 두 DataFrame을 병합하고 학습을 위한 컬럼을 처리합니다."""
    
    df_combined = pd.concat([df_real, df_virtual], ignore_index=True)
    df_combined = df_combined.drop_duplicates().reset_index(drop=True)
    
    # A.py의 파생 변수 계산 로직
    df_combined['T_Weld'] = df_combined['T_Melt'] * 0.8 + df_combined['T_Mold'] * 0.2 + df_combined['V_Inj'] * 0.1
    df_combined['t_Fill'] = 3.0 - 0.015 * df_combined['V_Inj']
    
    # A.py의 모델 구조를 위해 필요한 컬럼만 선택
    required_cols = FEATURES + [TARGET_VAR]
    df_combined = df_combined[[col for col in required_cols if col in df_combined.columns]].dropna()
    
    return df_combined

@st.cache_resource
def train_model(df):
    """모델을 학습하고 평가합니다. (A.py처럼 스케일링 없음)"""
    
    X = df[FEATURES]
    y = df[TARGET_VAR]

    if len(y.unique()) < 2:
        st.error(f"🚨 치명적 오류: 학습 데이터에 불량(1) 샘플이 부족합니다. 현재 불량률: {df[TARGET_VAR].mean()*100:.1f}%.")
        raise ValueError("불량 샘플이 부족합니다.")

    # A.py와 같이 스케일링 없이 모델 학습
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, FEATURES, accuracy, len(df), df[TARGET_VAR].mean()

def predict_weld_line_risk(model, input_data):
    """Weld Line 불량 위험 확률을 예측합니다. (스케일링 없음)"""
    
    df_input = pd.DataFrame([input_data])
    
    # 파생 변수 계산
    df_input['T_Weld'] = df_input['T_Melt'] * 0.8 + df_input['T_Mold'] * 0.2 + df_input['V_Inj'] * 0.1
    df_input['t_Fill'] = 3.0 - 0.015 * df_input['V_Inj']
    
    df_input = df_input[FEATURES] # A.py의 피처만 사용
    
    # 위험 확률 계산 (로지스틱 함수)
    risk_prob = model.predict_proba(df_input)[:, 1][0]
    prediction = 1 if risk_prob > 0.5 else 0

    return risk_prob, prediction

def find_optimal_conditions(model, initial_guess):
    """최적 공정 조건을 찾습니다. (A.py 기반으로 단순화)"""
    
    opt_var_names = PROCESS_VARS
    bounds = [(A_VARIABLE_BOUNDS[var][0], A_VARIABLE_BOUNDS[var][1]) for var in opt_var_names]
    
    def objective_function(X_opt, model):
        # 최적화 변수 설정
        T_Melt, V_Inj, P_Pack, T_Mold, Meter, VP_Switch_Pos = X_opt
        
        # 파생 변수 계산
        T_Weld = T_Melt * 0.8 + T_Mold * 0.2 + V_Inj * 0.1
        t_Fill = 3.0 - 0.015 * V_Inj
        
        input_data = {
            'T_Melt': T_Melt, 'V_Inj': V_Inj, 'P_Pack': P_Pack, 'T_Mold': T_Mold,
            'Meter': Meter, 'VP_Switch_Pos': VP_Switch_Pos, 'T_Weld': T_Weld, 't_Fill': t_Fill
        }
        
        df_input = pd.DataFrame([input_data])[FEATURES]

        # 위험 확률 (최소화 목표)
        risk_prob = model.predict_proba(df_input)[:, 1][0]
        
        return risk_prob

    result = minimize(
        objective_function, 
        initial_guess, 
        args=(model,),
        method='SLSQP',
        bounds=bounds,
        tol=1e-6
    )
    
    optimal_conditions = dict(zip(opt_var_names, result.x))
    optimal_risk = result.fun * 100
    
    return optimal_conditions, optimal_risk, result.success, result.message

def run_sensitivity_analysis(model, current_input):
    """주요 세 변수에 대한 민감도 분석을 수행합니다. (A.py 기반으로 단순화)"""
    
    analysis_results = {}
    variables_to_sweep = {
        'T_Melt': {'min': A_VARIABLE_BOUNDS['T_Melt'][0], 'max': A_VARIABLE_BOUNDS['T_Melt'][1], 'steps': 20, 'unit': '°C', 'label': '용융 온도 (T_Melt)'},
        'V_Inj': {'min': A_VARIABLE_BOUNDS['V_Inj'][0], 'max': A_VARIABLE_BOUNDS['V_Inj'][1], 'steps': 20, 'unit': 'mm/s', 'label': '사출 속도 (V_Inj)'},
        'T_Mold': {'min': A_VARIABLE_BOUNDS['T_Mold'][0], 'max': A_VARIABLE_BOUNDS['T_Mold'][1], 'steps': 20, 'unit': '°C', 'label': '금형 온도 (T_Mold)'}
    }

    base_input = current_input.copy() 

    for var_name, config in variables_to_sweep.items():
        sweep_values = np.linspace(config['min'], config['max'], config['steps'])
        risks = []
        
        for val in sweep_values:
            temp_input = base_input.copy()
            temp_input[var_name] = val
            
            # 위험도 예측
            risk_prob, _ = predict_weld_line_risk(model, temp_input)
            risks.append(risk_prob * 100) # 퍼센트로 저장
            
        analysis_results[var_name] = pd.DataFrame({
            var_name: sweep_values, 
            'Weld_Risk (%)': risks
        })
        analysis_results[var_name].rename(columns={var_name: config['label']}, inplace=True)

    return analysis_results

# =================================================================
# 2. STREAMLIT UI 및 세션 관리 (B.py의 GUI 구조 채택)
# =================================================================

def set_initial_vals():
    """A.py의 기본값을 세션 상태에 설정합니다."""
    st.session_state['initial_values'] = A_DEFAULT_INPUT_VALS.copy()

def load_and_train_model(uploaded_virtual, uploaded_real):
    """파일을 로드하고 모델을 학습합니다."""

    # ⭐️ 수정된 필수 파일 확인 로직: 2번 또는 3번 파일 중 하나라도 있으면 진행 ⭐️
    if uploaded_real is None and uploaded_virtual is None:
        st.error("🚨 필수 파일 경고: AI 모델 학습을 위해 최소한 **가상 데이터 또는 시뮬레이션 데이터 파일** 중 하나를 업로드해야 합니다.")
        st.session_state['model_loaded'] = False
        return

    with st.spinner('데이터 처리 및 AI 모델 학습 중...'):
        
        # 1. 파일 로드 및 기본값 처리
        df_real = load_df_from_uploader(uploaded_real)
        df_virtual = load_df_from_uploader(uploaded_virtual)
        
        if df_real is None: df_real = pd.DataFrame()
        if df_virtual is None: df_virtual = pd.DataFrame()
        
        set_initial_vals()
        
        # 2. 데이터 병합 및 처리
        st.session_state['df_weld'] = process_weld_data(df_virtual, df_real)
        st.session_state['virtual_data_size'] = len(df_virtual)
        st.session_state['real_data_size'] = len(df_real)
        
        # 3. 학습 가능성 확인
        if len(st.session_state['df_weld']) < 10: 
            st.error(f"🚨 학습 데이터가 너무 작습니다. 현재 데이터 크기: {len(st.session_state['df_weld'])}개. 최소 10개 이상을 권장합니다.")
            st.session_state['model_loaded'] = False
            return
        
        # 4. 모델 학습
        try:
            st.cache_resource.clear() 
            st.session_state['model'], st.session_state['feature_names'], st.session_state['accuracy'], st.session_state['data_size'], st.session_state['defect_rate'] = train_model(st.session_state['df_weld'])
            st.session_state['model_loaded'] = True
            st.session_state['executed'] = False 
            st.session_state['optimal_executed'] = False 
            st.success("✅ AI 모델 학습 및 로드 완료! 초기 조건이 UI에 반영되었습니다.")
        except ValueError as e:
             st.session_state['model_loaded'] = False
             st.error(f"모델 학습 실패: {e}")
        except Exception as e:
            st.session_state['model_loaded'] = False
            st.error(f"모델 학습 중 예기치 않은 오류 발생: {e}")

def run_optimization():
    if not st.session_state.get('model_loaded', False):
        st.error("AI 모델이 로드되지 않았습니다. 먼저 모델을 학습시켜 주세요.")
        st.session_state['optimal_executed'] = False
        return

    try:
        # 1. 현재 UI 공정 조건 사용
        initial_guess = [
            st.session_state['T_Melt_slider'],
            st.session_state['V_Inj_slider'],
            st.session_state['P_Pack_slider'],
            st.session_state['T_Mold_slider'],
            st.session_state['Meter_slider'],
            st.session_state['VP_Switch_Pos_slider']
        ]

    except KeyError as e:
        st.error(f"UI 입력값을 가져오는 중 오류 발생: {e}.")
        st.session_state['optimal_executed'] = False
        return

    model = st.session_state['model']
    
    with st.spinner('✨ 최적 조건 탐색 중...'):
        opt_cond, opt_risk, success, message = find_optimal_conditions(model, initial_guess)
        
    # 최적 결과 저장
    if success:
        st.session_state['optimal_conditions'] = opt_cond
        st.session_state['optimal_risk'] = opt_risk
        st.session_state['optimal_executed'] = True
        st.session_state['optimal_success'] = True
    else:
        st.session_state['optimal_executed'] = True
        st.session_state['optimal_success'] = False
        st.session_state['optimal_message'] = message


# --- 사이드바 ---
with st.sidebar:
    st.title("📂 데이터 파일 업로드 및 모델 학습")
    st.info("AI 모델 학습을 위해 최소한 **가상 데이터 또는 시뮬레이션 데이터 파일** 중 하나는 업로드해야 합니다.")
    
    # 파일 업로더
    st.file_uploader("1. 가상 학습 데이터 (test_condition.xlsx) [학습 데이터]", type=['xlsx', 'csv'], key='virtual_uploader')
    st.file_uploader("2. 시뮬레이션 학습 데이터 (moldflow_condition.xlsx) [학습 데이터]", type=['xlsx', 'csv'], key='real_uploader')
    
    # 로드 및 학습 버튼
    st.button(
        "🚀 파일 로드 및 AI 모델 학습 시작", 
        on_click=lambda: load_and_train_model(st.session_state.get('virtual_uploader'), st.session_state.get('real_uploader')),
        use_container_width=True, 
        type='primary'
    )
    
    st.markdown("---")
    
    st.subheader("시스템 상태")
    if st.session_state.get('model_loaded', False):
        st.markdown(f"""
        --- 모델: Weld Line 불량 예측 모델 (A.py 기반) ---
        **정확도 (Accuracy):** {st.session_state['accuracy']:.4f}
        **전체 데이터 수:** {st.session_state['data_size']}개, **불량률:** {st.session_state['defect_rate']*100:.1f}%
        **시뮬레이션 데이터:** {st.session_state.get('real_data_size', 'N/A')}개
        **가상 데이터:** {st.session_state.get('virtual_data_size', 'N/A')}개
        """)
    else:
        st.warning("파일을 업로드하고 'AI 모델 학습 시작' 버튼을 눌러주세요.")


if not st.session_state.get('model_loaded', False):
    st.error("데이터 파일이 업로드되고 AI 모델이 학습될 때까지 시스템을 사용할 수 없습니다.")
    st.stop() 

if 'initial_values' not in st.session_state:
    set_initial_vals()
    
initial_vals = st.session_state['initial_values'] 

# 탭 (B.py 구조)
tab1, tab2, tab3 = st.tabs(["1. Weld Line 공정 진단 (핵심)", "2. 모델 및 데이터 검토", "3. 민감도 분석"])

with tab1:
    st.subheader("A. 현재 공정 조건 입력")
    
    # --- 공정 변수 레이아웃 분리 (B.py 스타일) ---
    col_proc_temp, col_proc_dim = st.columns(2)
    
    bounds = A_VARIABLE_BOUNDS

    # 1. Process Condition (T_Melt, V_Inj, P_Pack, T_Mold)
    with col_proc_temp:
        st.markdown("##### ⚙️ 주요 온도/압력/속도 조건")
        col1, col2 = st.columns(2)
        T_Melt = col1.slider("1. 용융 온도 (T_Melt, °C)", bounds['T_Melt'][0], bounds['T_Melt'][1], int(initial_vals['T_Melt']), bounds['T_Melt'][2], key='T_Melt_slider')
        T_Mold = col2.slider("2. 금형 온도 (T_Mold, °C)", bounds['T_Mold'][0], bounds['T_Mold'][1], int(initial_vals['T_Mold']), bounds['T_Mold'][2], key='T_Mold_slider')
        
        col3, col4 = st.columns(2)
        V_Inj = col3.slider("3. 사출 속도 (V_Inj, mm/s)", bounds['V_Inj'][0], bounds['V_Inj'][1], int(initial_vals['V_Inj']), bounds['V_Inj'][2], key='V_Inj_slider')
        P_Pack = col4.slider("4. 보압 (P_Pack, MPa)", bounds['P_Pack'][0], bounds['P_Pack'][1], int(initial_vals['P_Pack']), bounds['P_Pack'][2], key='P_Pack_slider')

    # 2. Dimension Condition (Meter, VP_Switch_Pos)
    with col_proc_dim:
        st.markdown("##### 📐 계량 및 절환 위치")
        Meter = st.slider("5. 계량 거리 (Meter, mm)", bounds['Meter'][0], bounds['Meter'][1], float(initial_vals['Meter']), bounds['Meter'][2], key='Meter_slider')
        VP_Switch_Pos = st.slider("6. VP 절환 위치 (VP_Switch_Pos, mm)", bounds['VP_Switch_Pos'][0], bounds['VP_Switch_Pos'][1], float(initial_vals['VP_Switch_Pos']), bounds['VP_Switch_Pos'][2], key='VP_Switch_Pos_slider')
        
        # A.py에는 없는 섹션이므로 대체
        st.markdown("##### ℹ️ 진단 추가 정보 (A.py 기본 모델)")
        st.info("이 모델은 **노하우(Know-how)**나 **노하우 영향 계수**를 사용하지 않고, 6개의 핵심 공정 변수만을 사용하여 불량 위험을 진단합니다.")

    st.markdown("---")
    
    # --- C. 진단 실행 및 결과 (B.py 스타일) ---
    st.subheader("C. 진단 실행 및 결과")
    
    T_Weld = T_Melt * 0.8 + T_Mold * 0.2 + V_Inj * 0.1
    t_Fill = 3.0 - 0.015 * V_Inj
    
    # A.py 모델 입력 데이터
    input_data = {
        'T_Melt': T_Melt, 'V_Inj': V_Inj, 'P_Pack': P_Pack, 'T_Mold': T_Mold,
        'Meter': Meter, 'VP_Switch_Pos': VP_Switch_Pos, 'T_Weld': T_Weld, 't_Fill': t_Fill,
    }
    
    col_diag_btn, col_opt_btn = st.columns(2)
    
    # 진단 실행
    with col_diag_btn:
        if st.button("🔴 현재 조건 위험도 진단", use_container_width=True, type='primary'):
            model = st.session_state['model']
            
            risk_prob, prediction = predict_weld_line_risk(model, input_data)
            st.session_state['risk_prob'] = risk_prob
            st.session_state['prediction'] = prediction
            st.session_state['executed'] = True
            st.session_state['current_input_for_sensitivity'] = input_data
            
    # 최적화 실행
    with col_opt_btn:
        st.button(
            "✨ 최적 공정 조건 제시", 
            use_container_width=True, 
            type='secondary',
            on_click=run_optimization,
            help="Weld Line 불량 위험을 최소화하는 최적 공정 조건을 탐색합니다."
        )
    
    st.markdown("---")
    
    col_diag_res, col_opt_res = st.columns(2)

    with col_diag_res:
        st.markdown("##### 💡 현재 공정 진단 결과")
        if st.session_state.get('executed', False):
            risk_prob = st.session_state['risk_prob']
            
            if risk_prob > 0.5:
                st.error(f"🔴 AI 모델 경고! 불량 위험 확률: **{risk_prob*100:.1f}%**", icon="🚨")
                st.warning("현재 공정 조건은 위험도가 높습니다. **최적 조건 제시**를 통해 개선 방안을 확인하세요.")
                
            else:
                st.success(f"✅ 현재 조건 양호. (AI 예측 위험도: **{risk_prob*100:.1f}%**)", icon="👍")
        else:
            st.info("진단이 실행되지 않았습니다. '🔴 현재 조건 위험도 진단' 버튼을 눌러주세요.")

    with col_opt_res:
        st.markdown("##### ✨ 최적 조건 솔루션")
        if st.session_state.get('optimal_executed', False):
            if st.session_state['optimal_success']:
                opt_cond = st.session_state['optimal_conditions']
                opt_risk = st.session_state['optimal_risk']
                
                st.success(f"탐색 완료! 최소 위험 확률: **{opt_risk:.2f}%**")
                
                # 최적 조건 포맷팅
                opt_df = pd.DataFrame({
                    '변수': PROCESS_VARS,
                    '최적 값': [
                        f"{opt_cond['T_Melt']:.0f} °C", 
                        f"{opt_cond['V_Inj']:.0f} mm/s", 
                        f"{opt_cond['P_Pack']:.0f} MPa", 
                        f"{opt_cond['T_Mold']:.0f} °C", 
                        f"{opt_cond['Meter']:.2f} mm", 
                        f"{opt_cond['VP_Switch_Pos']:.2f} mm"
                    ]
                })
                st.dataframe(opt_df, hide_index=True)
                
            else:
                st.warning(f"최적화 계산 실패. 오류 메시지: {st.session_state.get('optimal_message', '알 수 없는 오류')}")
        else:
            st.info("'✨ 최적 공정 조건 제시' 버튼을 눌러 최소 위험 조건을 찾아보세요.")


with tab2:
    st.header("상세 모델 학습 결과 및 데이터 미리보기")
    
    st.subheader("AI 모델 학습 요약")
    st.markdown("AI 모델은 **로지스틱 회귀 (Logistic Regression)** 모델을 사용하여 학습되었습니다.")
    st.metric(label="AI 모델 정확도 (테스트 세트)", value=f"{st.session_state['accuracy'] * 100:.2f}%")
    st.metric(label="통합 데이터 총 크기", value=f"{st.session_state['data_size']}개")
    st.metric(label="통합 데이터 세트 불량률", value=f"{st.session_state['defect_rate'] * 100:.1f}%")
    
    st.markdown("---")
    
    st.subheader("모델 계수 시각화")
    if 'model' in st.session_state and 'feature_names' in st.session_state:
        model = st.session_state['model']
        feature_names = st.session_state['feature_names']
        
        coef_df = pd.DataFrame({
            '특징 (Feature)': feature_names,
            '계수 (Coefficient)': model.coef_[0]
        })
        
        st.dataframe(coef_df.sort_values(by='계수 (Coefficient)', ascending=False), height=400)
        st.caption("계수의 절댓값이 클수록 불량 위험 확률에 미치는 영향이 큽니다. 양수(+)는 위험 증가, 음수(-)는 위험 감소를 의미합니다. **참고: 이 모델은 데이터 스케일링을 적용하지 않았으므로, 계수 크기를 직접 비교하는 것은 주의가 필요합니다.**")


with tab3:
    st.header("민감도 분석 📊")
    st.info("현재 설정된 공정 조건을 기준으로, 주요 변수 변화에 따른 Weld Line 불량 위험 확률 변화를 분석합니다. 분석을 시작하기 전에 **'1. Weld Line 공정 진단 (핵심)' 탭에서 진단 실행**이 필요합니다.")
    
    if st.session_state.get('model_loaded', False) and st.session_state.get('executed', False):
        
        base_input = st.session_state['current_input_for_sensitivity']
        
        # 분석 실행
        with st.spinner('민감도 분석 시뮬레이션 중...'):
            analysis_results = run_sensitivity_analysis(
                st.session_state['model'], 
                base_input
            )
        
        st.success("민감도 분석 완료! 현재 공정 변수들의 위험 변화 곡선을 확인하세요.")

        # 시각화 (B.py 스타일)
        variables_to_sweep = {
            'T_Melt': {'label': '용융 온도 (T_Melt)', 'unit': '°C'},
            'V_Inj': {'label': '사출 속도 (V_Inj)', 'unit': 'mm/s'},
            'T_Mold': {'label': '금형 온도 (T_Mold)', 'unit': '°C'}
        }
        
        col_t_melt, col_v_inj = st.columns(2)
        col_t_mold, col_empty = st.columns(2)
        
        plot_cols = {
            'T_Melt': col_t_melt, 
            'V_Inj': col_v_inj, 
            'T_Mold': col_t_mold
        }

        for var_name, config in variables_to_sweep.items():
            df_plot = analysis_results[var_name]
            current_val = base_input[var_name]

            with plot_cols[var_name]:
                st.markdown(f"##### {config['label']}에 대한 민감도 분석")
                
                # Streamlit 기본 차트 사용
                st.line_chart(df_plot, x=config['label'], y='Weld_Risk (%)')
                
                # 현재 값 표시
                current_risk = df_plot.loc[df_plot[config['label']].round(1) == current_val.round(1), 'Weld_Risk (%)'].iloc[0]
                st.caption(f"빨간 점: 현재 입력 조건 ({current_val:.2f} {config['unit']}, 위험도: {current_risk:.2f}%)")

    else:
        st.warning("⚠️ 민감도 분석을 위해 **'1. Weld Line 공정 진단 (핵심)' 탭에서 공정 조건을 설정하고 '🔴 현재 조건 위험도 진단' 버튼**을 먼저 눌러주세요.")
