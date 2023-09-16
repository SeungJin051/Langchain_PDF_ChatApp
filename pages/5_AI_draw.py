import os
#  streamlit = Python에서 GUI 생성
import streamlit as st
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound

from tempfile import NamedTemporaryFile

from gtts import gTTS
from dotenv import load_dotenv # OPEN_API_KEY

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

# PyPDF2 = streamlit의 PDF 업로드를 읽기 위해 
from PyPDF2 import PdfReader

# langchain.text_splitter = PyPDF2의 텍스트를 chunks로 나눔
from langchain.text_splitter import RecursiveCharacterTextSplitter
# openAI의 embedding = 계산하고 반환
from langchain.embeddings.openai import OpenAIEmbeddings
# VectorStore = FAISS, Chroma X = VectorStore에서 duckdb.DuckDBPyConnection 접근 불가능
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain # 답변

import side_bar

# .env 파일로부터 환경 변수 로드
load_dotenv()

# 사이드 바 생성
side_bar.run_side_bar()

# 스트림릿 앱 헤더 설정
st.header("AI Tory와 대화하기! 💬")


DrawButton = st.button("그려줘")
# "그려줘" 버튼을 눌렀을 때 대화 내용 출력
if DrawButton:
    if 'generated' not in st.session_state or not st.session_state['generated']:
        st.warning("저장된 데이터가 없습니다.")
    else:
        conversation = ""
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            conversation += f"사용자: {st.session_state['past'][i]}\n"
            conversation += f"AI Tory: {st.session_state['generated'][i]}\n\n"
        st.write(conversation)

