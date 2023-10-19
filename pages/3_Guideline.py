import streamlit as st
from google.cloud import firestore
import pandas as pd
from side_bar import run_side_tap_history, set_bg_hack
import os
import openai

from dotenv import load_dotenv # OPEN_API_KEY
tory_image_path = "pages/images/tory.png" 

run_side_tap_history()
main_bg_ext = "pages/images/tory_back.png"

set_bg_hack(main_bg_ext)
tab1, tab2 = st.tabs(["AI Tory 가이드라인", "기록"])

with tab1 :
    st.title(':blue[AI Tory] 🤖')
    st.header('| ChatGPT 기반의 인공지능 동화 스토리봇')
    st.header('', divider='gray')
    st.subheader("기술스택")
    python_badge_url = "https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"
    streamlit_badge_url = "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"
    openai_badge_url = "https://img.shields.io/badge/OpenAI-412991.svg?&style=for-the-badge&logo=openai&logoColor=white"
    firebase_badge_url = "https://img.shields.io/badge/Firebase-FFCA28.svg?&style=for-the-badge&logo=firebase&logoColor=white"

    # 이미지를 중앙으로 정렬하는 HTML 및 CSS를 사용합니다.
    centered_image_html = f"""
        <div style="display: flex; justify-content: center; padding-bottom: 20px">
            <img src="{python_badge_url}" alt="Python Badge" style="max-width: 100%; padding: 20px;"/>
            <img src="{streamlit_badge_url}" alt="Streamlit Badge" style="max-width: 100%; padding: 20px;"/>
            <img src="{openai_badge_url}" alt="OpenAI Badge" style="max-width: 100%; padding: 20px;"/>
            <img src="{firebase_badge_url}" alt="Firebase Badge" style="max-width: 100%; padding: 20px;"/>
        </div>
    """
    st.write(centered_image_html, unsafe_allow_html=True)

    iframe_url = "https://scribehow.com/shared/How_to_Use_AI_Assistants_for_Various_Tasks__D2zG6K-bT-yfWuyVieL45A"
    st.markdown(f'<iframe src="{iframe_url}" width="100%" height="640" allowfullscreen frameborder="0"></iframe>', unsafe_allow_html=True)


with tab2:
    st.title("Tory 기록")

    # 파이어스토어 클라이언트 생성
    db = firestore.Client.from_service_account_json("pages/ai-tory-firebase-key.json")

    # ChatHistory 컬렉션 지정
    chat_collection_name = 'ChatHistory'
    role_collection_name = 'RoleHistory'

    # ChatHistory 데이터 가져오기 (chat-create-time 필드를 기준으로 내림차순으로 정렬)
    chat_data = db.collection(chat_collection_name).order_by("chat_create_time", direction=firestore.Query.DESCENDING).stream()

    # RoleHistory 데이터 가져오기 (role-create-time 필드를 기준으로 내림차순으로 정렬)
    role_data = db.collection(role_collection_name).order_by("role_create_time", direction=firestore.Query.DESCENDING).stream()

    # ChatHistory 데이터를 판다스 데이터프레임으로 변환
    chat_df_data = []
    for chat_doc in chat_data:
        chat_doc_data = chat_doc.to_dict()
        chat_df_data.append(chat_doc_data)

    # RoleHistory 데이터를 판다스 데이터프레임으로 변환
    role_df_data = []
    for role_doc in role_data:
        role_doc_data = role_doc.to_dict()
        role_df_data.append(role_doc_data)

    # ChatHistory 데이터프레임 생성
    chat_df = pd.DataFrame(chat_df_data)

    # RoleHistory 데이터프레임 생성
    role_df = pd.DataFrame(role_df_data)

    history1, history2 = st.tabs(["대화", "역할놀이"])

with history1:
    if st.button("ChatHistory 데이터 삭제"):
        # Firestore 컬렉션 참조
        chat_collection_ref = db.collection(chat_collection_name)
        
        # ChatHistory 컬렉션의 모든 문서 가져오기
        chat_documents = chat_collection_ref.stream()
        
        # ChatHistory 컬렉션의 모든 문서 삭제
        for chat_doc in chat_documents:
            chat_doc.reference.delete()
        
        # 삭제 완료 메시지 출력
        st.success("ChatHistory의 모든 데이터가 삭제되었습니다..")

    # ChatHistory 데이터를 출력
    st.table(chat_df.style.set_table_attributes('class="dataframe"'))

with history2:
    if st.button("RoleHistory 데이터 삭제"):
        # Firestore 컬렉션 참조
        role_collection_ref = db.collection(role_collection_name)
        
        # RoleHistory 컬렉션의 모든 문서 가져오기
        role_documents = role_collection_ref.stream()
        
        # RoleHistory 컬렉션의 모든 문서 삭제
        for role_doc in role_documents:
            role_doc.reference.delete()
        
        # 삭제 완료 메시지 출력
        st.success("RoleHistoy의 모든 데이터가 삭제되었습니다..")
        
    # RoleHistory 데이터를 출력
    # with st.expander('RoleHistory 데이터', expanded=True):
    # RoleHistory 테이블 표시
    st.table(role_df.style.set_table_attributes('class="dataframe"'))