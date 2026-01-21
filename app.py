import os
import time
import threading
import mysql.connector
import pandas as pd
import numpy as np
from flask import Flask, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

app = Flask(__name__)

# --- CẤU HÌNH ---
MODEL_FILE = "brain.h5"
MYSQL_CONFIG = {
    'host': 'gondola.proxy.rlwy.net',
    'user': 'root',
    'password': 'fkpFGoYtPMHBcewIJodnzIiQUZtNQxxc',
    'port': 28709,
    'database': 'railway'
}
LOOKBACK = 15
ai_brain = None

# --- CÁC HÀM XỬ LÝ BỘ NÃO TRÊN MYSQL ---

def init_db():
    """Tạo bảng lưu trữ model nếu chưa có"""
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_model_store (
            id INT PRIMARY KEY,
            model_data LONGBLOB,
            updated_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_brain_to_db():
    """Lưu file .h5 vào MySQL"""
    if not os.path.exists(MODEL_FILE): return
    with open(MODEL_FILE, "rb") as f:
        binary_data = f.read()
    conn = mysql.connector.connect(**MYSQL_CONFIG)
    cursor = conn.cursor()
    sql = "REPLACE INTO ai_model_store (id, model_data, updated_at) VALUES (1, %s, NOW())"
    cursor.execute(sql, (binary_data,))
    conn.commit()
    cursor.close()
    conn.close()
    print(">> [Database] Đã sao lưu bộ não lên MySQL.")

def load_brain_from_db():
    """Tải bộ não từ MySQL về máy ảo Railway"""
    global ai_brain
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT model_data FROM ai_model_store WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            with open(MODEL_FILE, "wb") as f:
                f.write(row[0])
            ai_brain = load_model(MODEL_FILE)
            print(">> [Database] Đã khôi phục bộ não thành công.")
            return True
    except:
        print(">> [Database] Chưa có bộ não cũ, sẽ học mới từ đầu.")
    return False

# --- QUY TRÌNH HỌC LIÊN TỤC ---

def continuous_learning_loop():
    """Vòng lặp tự học mỗi 30 phút"""
    global ai_brain
    while True:
        print("\n>> [AI] Bắt đầu chu kỳ tự cập nhật kiến thức...")
        try:
            conn = mysql.connector.connect(**MYSQL_CONFIG)
            df = pd.read_sql("SELECT result FROM taixiu_data ORDER BY sid DESC LIMIT 5000", conn)
            conn.close()
            
            data = df['result'].values[::-1].astype(float)
            if len(data) > LOOKBACK + 10:
                X, y = [], []
                for i in range(len(data) - LOOKBACK):
                    X.append(data[i : i + LOOKBACK])
                    y.append(data[i + LOOKBACK])
                X = np.array(X).reshape(-1, LOOKBACK, 1)
                y = np.array(y)

                if ai_brain is None:
                    model = Sequential([
                        Bidirectional(LSTM(64, return_sequences=True), input_shape=(LOOKBACK, 1)),
                        LSTM(32),
                        Dense(1, activation='sigmoid')
                    ])
                    model.compile(optimizer='adam', loss='binary_crossentropy')
                else:
                    model = ai_brain

                # Học nhanh 10 vòng
                model.fit(X, y, epochs=10, batch_size=64, verbose=0)
                model.save(MODEL_FILE)
                ai_brain = model
                
                # QUAN TRỌNG: Lưu ngay lên MySQL sau khi học xong
                save_brain_to_db()
                print(">> [AI] Đã học xong và đồng bộ lên Database.")
        except Exception as e:
            print(f">> [Lỗi Học] {e}")
        
        time.sleep(1800) # Nghỉ 30 phút rồi học tiếp

@app.route('/status')
def status():
    return jsonify({"status": "running", "model_ready": ai_brain is not None})

if __name__ == "__main__":
    init_db()
    load_brain_from_db() # Khôi phục não lúc khởi động
    
    # Chạy vòng lặp tự học ở luồng riêng (Background)
    threading.Thread(target=continuous_learning_loop, daemon=True).start()
    
    # Chạy API Server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
