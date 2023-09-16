#  streamlit = Python에서 GUI 생성
import streamlit as st
from dotenv import load_dotenv # OPEN_API_KEY

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

import side_bar
import openai

# .env 파일로부터 환경 변수 로드
load_dotenv()

# 사이드 바 생성
side_bar.run_side_bar()

# 스트림릿 앱 헤더 설정
st.header("AI Tory와 그림그리기")


DrawButton = st.button("그려줘")
# "그려줘" 버튼을 눌렀을 때 대화 내용 출력
if DrawButton:
    if 'generated' not in st.session_state or not st.session_state['generated']:
        st.warning("저장된 데이터가 없습니다.")
    else:
        conversation = ""
        for i in range(len(st.session_state['generated']) - 1, -1, -1):
            conversation += f"{st.session_state['generated'][i]}\n\n"
        st.write(conversation)
        user_input = conversation


        gpt_prompt = [{
            "role" : "system",
            "content" :  "Understand and paint Joaquin Soroya's artwork, soft natural light, {user_input}. Make it short."
        }]

        # gpt_prompt.append({
        #     "role" : "user",
        #     "content" : user_input
        # })

        with st.spinner("토리가 열심히 그림을 그리고 있어요.."):
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=gpt_prompt
            )

        prompt = gpt_response["choices"][0]["message"]["content"]
        st.write(prompt)

        dallE_response = openai.Image.create(
            prompt=prompt,
            size="1024x1024",
        )

        st.image(dallE_response["data"][0]["url"])
