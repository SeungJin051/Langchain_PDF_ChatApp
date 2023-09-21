import streamlit as st #  streamlit = Python에서 GUI 생성
from streamlit_extras.add_vertical_space import add_vertical_space

# 스트림릿의 사이드바 설정
def run_side_bar():
    with st.sidebar:
        st.title('🤖 AI Tory')
        st.markdown('''
        ## About
        - Notion [AI Tory](https://platform.openai.com/docs/models)
        - Github [AI Tory](https://python.langchain.com/)
        ''')
        st.sidebar.success('성공')
