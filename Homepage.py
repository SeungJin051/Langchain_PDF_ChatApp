import streamlit as st
import side_bar
st.title('Homepage')

side_bar.run_side_bar()
# https://scribehow.com/workspace#dashboard 수정 필요

# 외부 웹 페이지의 URL
iframe_url = "https://scribehow.com/embed/How_to_Use_AITory_to_Chat_Draw_and_Send_Files__T-W8Y4MsS4O1fSR_Oo8LvA"

with st.spinner('사용법 로딩...'):
    # iframe을 마크다운 형식으로 출력
    st.markdown(f'<iframe src="{iframe_url}" width="100%" height="640" allowfullscreen frameborder="0"></iframe>', unsafe_allow_html=True)