import streamlit as st #  streamlit = Python에서 GUI 생성
from streamlit_extras.add_vertical_space import add_vertical_space


# 스트림릿의 사이드바 설정
def run_side_bar():
    with st.sidebar:
        st.title('🤖 LLM AI Tory')
        st.markdown('''
        ## About
        AI Tory와 함께 놀아요! 😀
        - [OpenAI](https://platform.openai.com/docs/models) LLM model
        ''')

        add_vertical_space(5)
        st.write('Project [AI Tory](https://python.langchain.com/)')

        st.sidebar.success('성공')
