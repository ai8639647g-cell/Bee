import os
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from pypdf import PdfReader

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(
    page_title="Ultimate AI SuperApp",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. DATABASE
# ============================================================

DB_FILE = "ultimate_ai.db"


def init_database():
    conn = sqlite3.connect(DB_FILE)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def save_message(user_name, role, content):
    conn = sqlite3.connect(DB_FILE)

    conn.execute(
        """
        INSERT INTO conversations(user_name, role, content)
        VALUES (?, ?, ?)
        """,
        (user_name, role, content),
    )

    conn.commit()
    conn.close()


def load_messages(user_name, limit=50):
    conn = sqlite3.connect(DB_FILE)

    rows = conn.execute(
        """
        SELECT role, content
        FROM conversations
        WHERE user_name = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (user_name, limit),
    ).fetchall()

    conn.close()

    rows.reverse()
    return rows


def clear_database_messages(user_name):
    conn = sqlite3.connect(DB_FILE)

    conn.execute(
        "DELETE FROM conversations WHERE user_name = ?",
        (user_name,),
    )

    conn.commit()
    conn.close()


init_database()


# ============================================================
# 3. CYBERPUNK UI
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(circle at top right, #172554 0%, #020617 45%),
        #020617;
    color: #f8fafc;
}

[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid #1e3a8a;
}

h1, h2, h3 {
    letter-spacing: 0.5px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid #0891b2;
    background: linear-gradient(
        135deg,
        #0891b2,
        #2563eb
    );
    color: white;
    font-weight: 700;
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(14, 165, 233, 0.45);
}

.ai-card {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 16px;
    margin: 10px 0;
}

.user-card {
    border-left: 5px solid #06b6d4;
}

.bot-card {
    border-left: 5px solid #22c55e;
}

.code-card {
    background: #020617;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 12px;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    background: #172554;
    border: 1px solid #2563eb;
    margin-right: 5px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. SESSION STATE
# ============================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = DEFAULT_API_KEY

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "file_name" not in st.session_state:
    st.session_state.file_name = ""


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 AI CONTROL")

    st.caption("Ultimate AI SuperApp v2")

    st.divider()

    st.subheader("🔐 Gemini API")

    api_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
    )

    if api_key:
        st.session_state.api_key = api_key

    st.divider()

    st.subheader("👤 User Profile")

    user_name = st.text_input(
        "ชื่อของคุณ",
        value="ผู้ใช้",
    )

    user_preferences = st.text_area(
        "ความสนใจ",
        value="เขียนโปรแกรม, AI, Python, Web Development",
    )

    st.divider()

    st.subheader("⚙️ AI Settings")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1,
    )

    max_memory = st.slider(
        "จำนวนข้อความ Memory",
        5,
        100,
        30,
    )

    st.divider()

    if st.button("🗑️ ล้าง Chat Memory"):

        st.session_state.chat_history = []

        clear_database_messages(user_name)

        st.success("ล้าง Memory แล้ว")

    st.divider()

    st.caption("🔒 API Key ถูกใช้เฉพาะสำหรับการเรียก AI")


# ============================================================
# 6. CHECK API
# ============================================================

if not st.session_state.api_key:

    st.title("🌌 Ultimate AI SuperApp")

    st.warning(
        "กรุณาใส่ Gemini API Key ที่ Sidebar ก่อนใช้งาน AI"
    )

    st.stop()


# ============================================================
# 7. CREATE MODEL
# ============================================================

os.environ["GOOGLE_API_KEY"] = st.session_state.api_key

try:

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
    )

except Exception as e:

    st.error(f"ไม่สามารถสร้าง Gemini Model: {e}")

    st.stop()


# ============================================================
# 8. HELPER
# ============================================================

def call_ai(prompt, history=None):

    messages = []

    system_prompt = f"""
คุณคือ Ultimate AI Assistant

ข้อมูลผู้ใช้:
ชื่อ: {user_name}
ความสนใจ: {user_preferences}

หน้าที่:
- ตอบภาษาไทยเป็นหลัก
- ถ้าผู้ใช้ถามเรื่องโค้ด ให้ตอบแบบ Coding Assistant
- อธิบาย Error ให้เข้าใจง่าย
- ถ้าไม่แน่ใจ ห้ามสร้างข้อมูลขึ้นมาเอง
- ถ้าเป็นคำถามเกี่ยวกับไฟล์ ให้ใช้เฉพาะข้อมูลจากไฟล์
"""

    messages.append(
        SystemMessage(content=system_prompt)
    )

    if history:

        messages.extend(history)

    messages.append(
        HumanMessage(content=prompt)
    )

    response = llm.invoke(messages)

    return response.content


def split_text(text, chunk_size=1200):

    words = text.split()

    chunks = []

    current = []

    current_length = 0

    for word in words:

        current.append(word)
        current_length += len(word) + 1

        if current_length >= chunk_size:

            chunks.append(" ".join(current))

            current = []
            current_length = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def retrieve_chunks(text, query, top_k=4):

    chunks = split_text(text)

    query_words = set(
        re.findall(
            r"\w+",
            query.lower(),
            flags=re.UNICODE,
        )
    )

    scored = []

    for chunk in chunks:

        chunk_words = set(
            re.findall(
                r"\w+",
                chunk.lower(),
                flags=re.UNICODE,
            )
        )

        score = len(
            query_words.intersection(chunk_words)
        )

        scored.append((score, chunk))

    scored.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        chunk
        for score, chunk in scored[:top_k]
        if score > 0
    ]


# ============================================================
# 9. HEADER
# ============================================================

st.title("🌌 Ultimate AI SuperApp")

st.markdown(
    """
<span class="badge">🤖 Gemini</span>
<span class="badge">🧠 Memory</span>
<span class="badge">📁 RAG</span>
<span class="badge">📊 Machine Learning</span>
<span class="badge">💻 Coding AI</span>
""",
    unsafe_allow_html=True,
)

st.divider()


# ============================================================
# 10. TABS
# ============================================================

tabs = st.tabs(
    [
        "💬 AI Chat",
        "📁 RAG / Files",
        "📊 Machine Learning",
        "💻 Coding Assistant",
        "🧪 AI Lab",
    ]
)


# ============================================================
# TAB 1
# CHAT + MEMORY
# ============================================================

with tabs[0]:

    st.header("💬 AI Chat + Long-Term Memory")

    st.caption(
        f"สวัสดี {user_name} — AI รู้ว่าคุณสนใจ: {user_preferences}"
    )

    # Load DB history once
    if not st.session_state.chat_history:

        database_messages = load_messages(
            user_name,
            max_memory,
        )

        for role, content in database_messages:

            if role == "human":

                st.session_state.chat_history.append(
                    HumanMessage(content=content)
                )

            elif role == "ai":

                st.session_state.chat_history.append(
                    AIMessage(content=content)
                )

    # Display history
    for message in st.session_state.chat_history:

        if isinstance(message, HumanMessage):

            st.markdown(
                f"""
                <div class="ai-card user-card">
                <b>👤 {user_name}</b><br><br>
                {message.content}
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif isinstance(message, AIMessage):

            st.markdown(
                f"""
                <div class="ai-card bot-card">
                <b>🤖 Ultimate AI</b><br><br>
                {message.content}
                </div>
                """,
                unsafe_allow_html=True,
            )

    user_input = st.chat_input(
        "ถาม AI ได้เลย..."
    )

    if user_input:

        st.session_state.chat_history.append(
            HumanMessage(content=user_input)
        )

        save_message(
            user_name,
            "human",
            user_input,
        )

        history_for_ai = (
            st.session_state.chat_history[-max_memory:]
        )

        with st.spinner("🤖 AI กำลังคิด..."):

            try:

                answer = call_ai(
                    user_input,
                    history_for_ai[:-1],
                )

                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

                save_message(
                    user_name,
                    "ai",
                    answer,
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"เกิดข้อผิดพลาด: {e}"
                )


# ============================================================
# TAB 2
# FILE + RAG
# ============================================================

with tabs[1]:

    st.header("📁 Document Intelligence / RAG")

    st.write(
        "อัปโหลดเอกสาร แล้วถาม AI จากข้อมูลในเอกสาร"
    )

    uploaded_file = st.file_uploader(
        "เลือกไฟล์",
        type=[
            "txt",
            "csv",
            "json",
            "pdf",
        ],
    )

    if uploaded_file:

        try:

            file_bytes = uploaded_file.read()

            file_name = uploaded_file.name.lower()

            extracted_text = ""

            # TXT
            if file_name.endswith(".txt"):

                extracted_text = file_bytes.decode(
                    "utf-8",
                    errors="ignore",
                )

            # CSV
            elif file_name.endswith(".csv"):

                import io

                df = pd.read_csv(
                    io.BytesIO(file_bytes)
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                )

                extracted_text = df.to_string(
                    index=False
                )

            # JSON
            elif file_name.endswith(".json"):

                json_data = json.loads(
                    file_bytes.decode(
                        "utf-8",
                        errors="ignore",
                    )
                )

                st.json(json_data)

                extracted_text = json.dumps(
                    json_data,
                    ensure_ascii=False,
                    indent=2,
                )

            # PDF
            elif file_name.endswith(".pdf"):

                import io

                reader = PdfReader(
                    io.BytesIO(file_bytes)
                )

                pages = []

                for page in reader.pages:

                    text = page.extract_text()

                    if text:
                        pages.append(text)

                extracted_text = "\n".join(
                    pages
                )

            st.session_state.file_text = extracted_text
            st.session_state.file_name = uploaded_file.name

            st.success(
                f"อ่านไฟล์สำเร็จ: {uploaded_file.name}"
            )

            st.info(
                f"ขนาดข้อมูล: {len(extracted_text):,} ตัวอักษร"
            )

        except Exception as e:

            st.error(
                f"อ่านไฟล์ไม่ได้: {e}"
            )

    if st.session_state.file_text:

        st.divider()

        st.subheader(
            f"📄 {st.session_state.file_name}"
        )

        query = st.text_input(
            "ถามคำถามเกี่ยวกับเอกสาร"
        )

        if query:

            with st.spinner(
                "🔎 กำลังค้นหาข้อมูลที่เกี่ยวข้อง..."
            ):

                relevant_chunks = retrieve_chunks(
                    st.session_state.file_text,
                    query,
                )

                if not relevant_chunks:

                    st.warning(
                        "ไม่พบข้อมูลที่ตรงกับคำถาม"
                    )

                else:

                    context = "\n\n---\n\n".join(
                        relevant_chunks
                    )

                    rag_prompt = f"""
คุณกำลังตอบคำถามจากเอกสาร

ข้อมูลที่ค้นพบ:
----------------
{context}
----------------

คำถาม:
{query}

กฎ:
1. ตอบจากข้อมูลที่ให้เท่านั้น
2. ถ้าข้อมูลไม่เพียงพอ ให้บอกว่าไม่พบข้อมูล
3. ห้ามสร้างข้อมูลขึ้นมาเอง
4. ตอบภาษาไทย
"""

                    try:

                        answer = llm.invoke(
                            [
                                HumanMessage(
                                    content=rag_prompt
                                )
                            ]
                        )

                        st.success(
                            "📋 คำตอบจากเอกสาร"
                        )

                        st.write(
                            answer.content
                        )

                    except Exception as e:

                        st.error(
                            f"AI Error: {e}"
                        )


# ============================================================
# TAB 3
# MACHINE LEARNING
# ============================================================

with tabs[2]:

    st.header(
        "📊 Machine Learning Laboratory"
    )

    st.subheader(
        "🏠 Linear Regression"
    )

    st.write(
        "ทดลองฝึกโมเดลจากข้อมูลพื้นที่บ้านและราคา"
    )

    sample_data = pd.DataFrame(
        {
            "area": [
                50,
                80,
                100,
                120,
                150,
                180,
                200,
                250,
                300,
            ],
            "price": [
                1.0,
                1.5,
                2.0,
                2.4,
                3.0,
                3.6,
                4.0,
                5.0,
                6.0,
            ],
        }
    )

    st.dataframe(
        sample_data,
        use_container_width=True,
    )

    X = sample_data[["area"]]
    y = sample_data["price"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    )

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "MAE",
            f"{mae:.3f}",
        )

    with col2:

        st.metric(
            "R²",
            f"{r2:.3f}",
        )

    st.divider()

    area = st.number_input(
        "พื้นที่บ้าน (m²)",
        min_value=1.0,
        value=180.0,
    )

    prediction = model.predict(
        np.array([[area]])
    )[0]

    st.success(
        f"🤖 Model คาดการณ์ราคา ≈ {prediction:.2f} ล้านบาท"
    )

    # Plot
    fig, ax = plt.subplots()

    ax.scatter(
        sample_data["area"],
        sample_data["price"],
    )

    line_x = np.linspace(
        sample_data["area"].min(),
        sample_data["area"].max(),
        100,
    )

    line_y = model.predict(
        line_x.reshape(-1, 1)
    )

    ax.plot(
        line_x,
        line_y,
    )

    ax.set_xlabel(
        "Area (m²)"
    )

    ax.set_ylabel(
        "Price (Million Baht)"
    )

    ax.set_title(
        "Linear Regression"
    )

    st.pyplot(fig)


# ============================================================
# TAB 4
# CODING ASSISTANT
# ============================================================

with tabs[3]:

    st.header(
        "💻 AI Coding Assistant"
    )

    language = st.selectbox(
        "ภาษาโปรแกรม",
        [
            "Python",
            "JavaScript",
            "HTML",
            "CSS",
            "Lua",
            "SQL",
            "C++",
            "Java",
        ],
    )

    code = st.text_area(
        "วางโค้ดของคุณ",
        height=300,
        placeholder="วาง code ที่ต้องการให้ AI วิเคราะห์...",
    )

    task = st.selectbox(
        "ต้องการให้ AI ทำอะไร?",
        [
            "อธิบายโค้ด",
            "ค้นหา Error",
            "แก้ไขโค้ด",
            "ปรับปรุงโค้ด",
            "เพิ่ม Feature",
            "เขียนใหม่",
        ],
    )

    if st.button(
        "🚀 วิเคราะห์โค้ด"
    ):

        if not code.strip():

            st.warning(
                "กรุณาใส่โค้ดก่อน"
            )

        else:

            coding_prompt = f"""
คุณคือ Senior Software Engineer

ภาษา:
{language}

งาน:
{task}

โค้ด:
```{language.lower()}
{code}

ตอบเป็นภาษาไทย

ถ้ามี Error:

1. ระบุ Error
2. อธิบายสาเหตุ
3. แสดงโค้ดที่แก้แล้ว
4. อธิบายว่าทำไมโค้ดใหม่ถึงแก้ปัญหาได้

ถ้าไม่มี Error:
แนะนำวิธีปรับปรุงโค้ด
"""

        with st.spinner(
            "🧠 AI กำลังวิเคราะห์โค้ด..."
        ):

            try:

                response = llm.invoke(
                    [
                        HumanMessage(
                            content=coding_prompt
                        )
                    ]
                )

                st.markdown(
                    '<div class="code-card">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    response.content
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

            except Exception as e:

                st.error(
                    f"AI Error: {e}"
                )

============================================================

TAB 5

AI LAB

============================================================

with tabs[4]:

st.header(
    "🧪 AI Laboratory"
)

st.subheader(
    "💬 Sentiment Analysis"
)

review = st.text_area(
    "ข้อความที่ต้องการวิเคราะห์",
    value="สินค้าดีมาก ส่งเร็ว แต่แพ็กเกจเสียหายนิดหน่อย",
)

if st.button(
    "🔍 วิเคราะห์ Sentiment"
):

    sentiment_prompt = f"""

วิเคราะห์ข้อความนี้:

"{review}"

ให้ตอบ:

- Positive
- Negative
- Neutral

พร้อมเหตุผลสั้น ๆ

ตอบภาษาไทย
"""

    with st.spinner(
        "กำลังวิเคราะห์..."
    ):

        try:

            result = llm.invoke(
                [
                    HumanMessage(
                        content=sentiment_prompt
                    )
                ]
            )

            st.info(
                result.content
            )

        except Exception as e:

            st.error(
                str(e)
            )

st.divider()

st.subheader(
    "🧮 AI Calculator"
)

expression = st.text_input(
    "ใส่นิพจน์ เช่น 125*30+500"
)

if st.button(
    "คำนวณ"
):

    # จำกัด calculator ให้เป็นตัวเลขและ operator พื้นฐาน
    if re.fullmatch(
        r"[0-9+\-*/(). %]+",
        expression,
    ):

        try:

            result = eval(
                expression,
                {
                    "__builtins__": {}
                },
                {},
            )

            st.success(
                f"ผลลัพธ์ = {result}"
            )

        except Exception:

            st.error(
                "นิพจน์ไม่ถูกต้อง"
            )

    else:

        st.error(
            "อนุญาตเฉพาะตัวเลขและ + - * / ( ) %"
        )

============================================================

FOOTER

============================================================

st.divider()

st.caption(
"Ultimate AI SuperApp v2 • Streamlit + Gemini + RAG + Machine Learning"
)

:::

---

# 5. ติดตั้งบนคอมพิวเตอร์

### Windows

เปิด **Command Prompt / PowerShell**

เข้าโฟลเดอร์:

```bash
cd UltimateAI

สร้าง Virtual Environment:

python -m venv .venv

เปิดใช้งาน:

.venv\Scripts\activate

จากนั้นติดตั้ง:

pip install -r requirements.txt

รัน:

streamlit run app.py

แล้วจะได้ URL ประมาณ:

http://localhost:8501

เปิดด้วย Chrome/Edge ได้เลย

---

6. macOS / Linux

cd UltimateAI

python3 -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

แล้วเปิด:

http://localhost:8501

---

7. รันจากมือถือ Android 📱

มือถือ Android สามารถใช้ได้ แต่มี 2 วิธี

วิธีง่ายที่สุด

ให้ คอมพิวเตอร์เป็นเครื่องรัน AI แล้วมือถือเปิดเว็บผ่าน Wi-Fi เดียวกัน

รัน:

streamlit run app.py --server.address 0.0.0.0

จากนั้นดู IP ของคอม เช่น:

192.168.1.10

มือถือเปิด:

http://192.168.1.10:8501

มือถือกับคอมต้องอยู่ Wi-Fi เดียวกัน

ถ้า Windows Firewall ถาม ให้ Allow Python/Streamlit บน Private Network

---

8. ถ้าไม่มีคอม

สามารถเอาโปรเจกต์นี้ขึ้น Streamlit Community Cloud แล้วเปิดจาก Android ได้

โครงสร้าง GitHub:

UltimateAI
│
├── app.py
├── requirements.txt
└── .gitignore

แต่ อย่าใส่ ".env" เข้า GitHub

บน Streamlit Cloud ให้เก็บ API Key ใน Secrets แทน:

GEMINI_API_KEY = "YOUR_API_KEY"

ตัวแอปจะอ่านค่าจาก:

os.getenv("GEMINI_API_KEY")

---

9. ตอนนี้ AI ตัวนี้ทำอะไรได้บ้าง?

หลังรันแล้วจะมี 5 ส่วน:

🌌 Ultimate AI SuperApp
│
├── 💬 AI Chat
│    ├── Gemini
│    ├── Chat Memory
│    ├── User Profile
│    └── SQLite Database
│
├── 📁 RAG / Files
│    ├── TXT
│    ├── CSV
│    ├── JSON
│    └── PDF
│
├── 📊 Machine Learning
│    ├── Dataset
│    ├── Train/Test
│    ├── Linear Regression
│    ├── MAE
│    ├── R²
│    └── Graph
│
├── 💻 Coding Assistant
│    ├── Explain
│    ├── Debug
│    ├── Fix
│    ├── Refactor
│    └── Generate
│
└── 🧪 AI Lab
     ├── Sentiment
     └── Calculator

จุดสำคัญ: ส่วน RAG ในเวอร์ชันนี้เป็น RAG แบบง่ายที่ค้นหา chunk ด้วยคำที่ตรงกัน ยังไม่ได้ใช้ vector embeddings/FAISS ดังนั้นถ้าต้องการระบบ RAG ระดับจริง เราสามารถอัปเกรดต่อเป็น:

PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embeddings
 ↓
Vector Database
 ↓
Semantic Search
 ↓
Gemini
 ↓
คำตอบพร้อมแหล่งที่มา

และขั้นต่อไปสามารถเพิ่ม Vision + Voice + AI Agent + Vector Database + AI Coding Workspace ให้กลายเป็น AI OS ตัวเต็มได้

ถ้าคุณต้องการ ผมสามารถ"สร้าง ภาพไดอะแกรมแสดงการทำงานของ Ultimate AI SuperApp ทั้งระบบ ให้ด้วยครับ" (reference-followup:26046)
