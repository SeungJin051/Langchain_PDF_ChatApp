import streamlit as st
from side_bar import run_side_tap_home

tory_image_path = "pages/images/tory.png" 

run_side_tap_home()

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

iframe_url = "https://scribehow.com/embed/How_to_Use_AITory_to_Chat_Draw_and_Send_Files__T-W8Y4MsS4O1fSR_Oo8LvA"
st.markdown(f'<iframe src="{iframe_url}" width="100%" height="640" allowfullscreen frameborder="0"></iframe>', unsafe_allow_html=True)