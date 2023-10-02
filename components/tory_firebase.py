import streamlit as st
import side_bar
import datetime

from firebase_admin import firestore

def add_firebase_chat(query, output):
    # Firestore 데이터베이스에 연결합니다.
    db = firestore.Client.from_service_account_json("pages/ai-tory-firebase-key.json")
    # form을 사용하여 질문을 입력합니다.
        # 질문 버튼이 클릭되면 처리할 코드를 추가합니다.
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y년 %m월 %d일 %H시")

    # Firestore에 새로운 문서를 추가합니다.
    # 문서 ID를 현재 날짜와 시간으로 설정하여 중복을 방지합니다.
    doc_ref = db.collection("ChatHistory").add({
    # 질문과 AI의 대답을 문서에 추가합니다.
        "chat_user_question": query,
        "chat_ai_question": output,
        "chat_create_time": formatted_datetime
    })

def add_firebase_role(query, output):
    # Firestore 데이터베이스에 연결합니다.
    db = firestore.Client.from_service_account_json("pages/ai-tory-firebase-key.json")
    # form을 사용하여 질문을 입력합니다.
        # 질문 버튼이 클릭되면 처리할 코드를 추가합니다.
    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("%Y년 %m월 %d일 %H시")

    # Firestore에 새로운 문서를 추가합니다.
    # 문서 ID를 현재 날짜와 시간으로 설정하여 중복을 방지합니다.
    doc_ref = db.collection("RoleHistory").add({
    # 질문과 AI의 대답을 문서에 추가합니다.
        "role_user_question": query,
        "role_ai_question": output,
        "role_create_time": formatted_datetime
    })
