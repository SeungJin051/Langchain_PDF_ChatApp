import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter # langchain.text_splitter = PyPDF2의 텍스트를 chunks로 나눔
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle

# 스트림릿의 사이드바 설정
def run_side_bar():
     with st.sidebar:
        st.title('🤖 AI Tory')

        st.write('AI Tory에게 학습할 동화 PDF를 Upload 해주세요', expanded=True)
        pdf = st.file_uploader(label=' ', type='pdf', key='pdf') 

        pdf_reader = None
        text = ""
        VectorStore = None

        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            text = ""

            # 업로드한 PDF에서 텍스트 추출
            for page in pdf_reader.pages:
                text += page.extract_text()

            # 텍스트를 적절한 크기의 청크로 나누기
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text=text)

            # PDF 파일 이름으로 저장소 생성 또는 로드
            store_name = pdf.name[:-4]

            if os.path.exists(f"pdfs/{store_name}.pkl"):
                with open(f"pdfs/{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                print("해당 PDF는 저장소에 있습니다!")
            else:
                embedding = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embedding)
                with open(f"pdfs/{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                
                st.session_state['uploaded_pdf'] = pdf
                print("해당 PDF는 저장소에 없습니다!")
                st.success("성공")
        return pdf, text, VectorStore

def run_side_tap_draw():
     with st.sidebar:
        st.title('🤖 AI Tory')
        # 드롭다운 생성
        option = st.selectbox('원하는 항목을 선택하세요.', ('AI 그림 그리기', 'AI 대화 분석'))
        # 선택한 옵션 출력
        st.success("성공")
        
        return option
        
def run_side_tap_history():
     with st.sidebar:
        st.title('🤖 AI Tory')
        # 드롭다운 생성
        option = st.selectbox('원하는 항목을 선택하세요.', ('대시보드', '채팅', '역할놀이',))
        # 선택한 옵션 출력
        st.success("성공")
        
        return option