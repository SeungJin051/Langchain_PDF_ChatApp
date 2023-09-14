import streamlit as st
from google.cloud import firestore
import pandas as pd
import side_bar


side_bar.run_side_bar()

# 파이어스토어 클라이언트 생성
db = firestore.Client.from_service_account_json("pages/ai-tory-firebase-key.json")

# 파이어스토어 컬렉션 지정
collection_name = 'History'

# 스트림릿 애플리케이션 제목 설정
st.title('Firestore 데이터 테이블')

# 파이어스토어 데이터 가져오기
data = db.collection(collection_name).stream()

# 데이터를 판다스 데이터프레임으로 변환
df_data = []
for doc in data:
    doc_data = doc.to_dict()
    df_data.append(doc_data)

# 데이터프레임 생성
df = pd.DataFrame(df_data)

# 스타일 및 레이아웃 조절
st.markdown("""
<style>
    table {
        border-collapse: collapse;
        width: 100%;
    }


    th, td {
        text-align: left;
        padding: 20px;
    }

    tr:nth-child(even) {
        background-color: #212529;
    }

    th {
        background-color: #2C3034;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 테이블을 깔끔하게 디자인하고 높이 조절
st.table(df.style.set_table_attributes('class="dataframe"'))
# st.dataframe(df.style.set_table_attributes('class="dataframe"'))