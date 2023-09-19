import streamlit as st
from google.cloud import firestore
import pandas as pd
import side_bar

# empty 예외처리 

side_bar.run_side_bar()

# 파이어스토어 클라이언트 생성
db = firestore.Client.from_service_account_json("pages/ai-tory-firebase-key.json")

# ChatHistory 컬렉션 지정
chat_collection_name = 'ChatHistory'
role_collection_name = 'RoleHistory'

# 스트림릿 애플리케이션 제목 설정
st.title('기록된 데이터 테이블')
st.caption('토리와의 대화 기록이에요!')

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

# 스타일 및 레이아웃 조절
# st.markdown("""
# <style>
#     table {
#         border-collapse: collapse;
#         width: 100%;
#     }

#     th, td {
#         text-align: left;
#         padding: 20px;
#     }

#     tr:nth-child(even) {
#         background-color: #212529;
#     }

#     th {
#         background-color: #2C3034;
#         color: white;
#     }
# </style>
# """, unsafe_allow_html=True)

# ChatHistory 데이터를 출력
with st.expander('ChatHistory 데이터', expanded=True):
    # ChatHistory 테이블을 깔끔하게 디자인하고 높이 조절
    st.header('ChatHistory 데이터')

    # ChatHistory 테이블 표시
    st.table(chat_df.style.set_table_attributes('class="dataframe"'))

    # ChatHistory 컬렉션의 모든 문서를 삭제하는 버튼 추가
    if st.button("ChatHistory 데이터 모두 삭제"):
        # Firestore 컬렉션 참조
        chat_collection_ref = db.collection(chat_collection_name)
        
        # ChatHistory 컬렉션의 모든 문서 가져오기
        chat_documents = chat_collection_ref.stream()
        
        # ChatHistory 컬렉션의 모든 문서 삭제
        for chat_doc in chat_documents:
            chat_doc.reference.delete()
        
        # 삭제 완료 메시지 출력
        st.success("ChatHistory의 모든 데이터가 삭제되었습니다.. 새로고침을 해주세요.")


# RoleHistory 데이터를 출력
with st.expander('RoleHistory 데이터', expanded=True):
    # RoleHistory 테이블을 깔끔하게 디자인하고 높이 조절
    st.header('RoleHistory 데이터')
    
    # RoleHistory 테이블 표시
    st.table(role_df.style.set_table_attributes('class="dataframe"'))

    # RoleHistory 데이터 삭제 버튼 추가
    if st.button("RoleHistory 데이터 삭제"):
        # Firestore 컬렉션 참조
        role_collection_ref = db.collection(role_collection_name)
        
        # RoleHistory 컬렉션의 모든 문서 가져오기
        role_documents = role_collection_ref.stream()
        
        # RoleHistory 컬렉션의 모든 문서 삭제
        for role_doc in role_documents:
            role_doc.reference.delete()
        
        # 삭제 완료 메시지 출력
        st.success("RoleHistoy의 모든 데이터가 삭제되었습니다.. 새로고침을 해주세요.")
        