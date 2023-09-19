import streamlit as st
import side_bar
st.title('Homepage')

side_bar.run_side_bar()

python_badge_url = "https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"
streamlit_badge_url = "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"
openai_badge_url = "https://img.shields.io/badge/OpenAI-412991.svg?&style=for-the-badge&logo=openai&logoColor=white"
fireabse_badge_url = "https://img.shields.io/badge/Firebase-FFCA28.svg?&style=for-the-badge&logo=firebase&logoColor=white"

# "https://img.shields.io/badge/기술명-원하는색상코드.svg?&style=for-the-badge&logo=로고명&logoColor=로고색상"
# 파이썬, GitHub, 랭체인, 스트림릿 뱃지를 1줄로 정렬하여 스트림릿 앱에 추가
st.image([python_badge_url, streamlit_badge_url, openai_badge_url, fireabse_badge_url])

# https://scribehow.com/workspace#dashboard 수정 필요
# 외부 웹 페이지의 URL
iframe_url = "https://scribehow.com/embed/How_to_Use_AITory_to_Chat_Draw_and_Send_Files__T-W8Y4MsS4O1fSR_Oo8LvA"

with st.spinner('사용법 로딩...'):
    # iframe을 마크다운 형식으로 출력
    st.markdown(f'<iframe src="{iframe_url}" width="100%" height="640" allowfullscreen frameborder="0"></iframe>', unsafe_allow_html=True)