
import streamlit as st
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# ฝึกโมเดลฝังไว้ในระบบ
@st.cache_resource
def train_model():
    import pandas as pd
    np.random.seed(42)
    results = np.random.choice(['P', 'B', 'T'], size=3000, p=[0.45, 0.45, 0.1])
    df = pd.DataFrame({'result': results})
    result_map = {'P': 0, 'B': 1, 'T': 2}
    df['result_code'] = df['result'].map(result_map)

    for i in range(1, 6):
        df[f'prev_{i}'] = df['result_code'].shift(i)

    def get_streak(data):
        streaks = []
        current = None
        count = 0
        for val in data:
            if val == current:
                count += 1
            else:
                count = 1
                current = val
            streaks.append(count)
        return streaks

    df['streak'] = get_streak(df['result_code'])

    def detect_pattern(p1, p2, p3, p4, p5):
        pattern = [p1, p2, p3, p4, p5]
        if pattern == pattern[::-1]: return 'mirror'
        if all(p == pattern[0] for p in pattern): return 'dragon'
        if len(set(pattern)) == 2 and pattern[::2] == pattern[::2][::-1]: return 'pingpong'
        return 'mixed'

    df['pattern_type'] = df.apply(lambda row: detect_pattern(row['prev_1'], row['prev_2'], row['prev_3'], row['prev_4'], row['prev_5']), axis=1)
    df.dropna(inplace=True)

    X = df[['prev_1', 'prev_2', 'prev_3', 'prev_4', 'prev_5', 'streak']]
    y = df['result_code']
    patterns = df['pattern_type']

    le = LabelEncoder()
    X['pattern_type'] = le.fit_transform(patterns)

    model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)
    model.fit(X, y)
    return model, le

model, encoder = train_model()

# Mapping
label_map = {'P': 0, 'B': 1, 'T': 2}
reverse_map = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}

def detect_pattern(p):
    if p == p[::-1]: return 'mirror'
    if all(i == p[0] for i in p): return 'dragon'
    if len(set(p)) == 2 and p[::2] == p[::2][::-1]: return 'pingpong'
    return 'mixed'

def get_streak(seq):
    count = 1
    for i in range(len(seq)-2, -1, -1):
        if seq[i] == seq[-1]:
            count += 1
        else:
            break
    return count

# UI
st.title("Baccarat Predictor (เวอร์ชันใช้ง่าย)")
st.write("พิมพ์ผล 5 ตาหลังสุด เช่น: `P B B P P` หรือ `B B B B B`")

input_text = st.text_input("ผล 5 ตาหลังสุด (คั่นด้วยช่องว่าง)").strip().upper()

if input_text:
    tokens = input_text.split()
    if len(tokens) != 5 or any(t not in label_map for t in tokens):
        st.error("กรุณาพิมพ์ให้ถูกต้อง เช่น: `P B B P P`")
    else:
        code_seq = [label_map[t] for t in tokens]
        pattern = detect_pattern(code_seq)
        streak = get_streak(code_seq)
        pattern_code = encoder.transform([pattern])[0]
        features = code_seq + [streak, pattern_code]
        pred = model.predict([features])[0]
        st.success(f"ระบบคาดการณ์ว่า: **{reverse_map[pred]}**")
