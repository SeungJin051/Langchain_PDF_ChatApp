import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import openai
import googletrans

from side_bar import run_side_tap_draw
from dotenv import load_dotenv # OPEN_API_KEY

# -------

# .env 파일로부터 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 사이드 바 생성
run_side_tap_draw()


# 스트림릿 앱 헤더 설정
st.header("AI Tory의 그림 그리기 🎨")
st.caption('토리가 학습한 정보로 그림을 그려줄게요 🔥')

st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get('text') or st.session_state.text == []:
    st.warning("토리와 대화를 먼저 해주세요.")

# st.write(st.session_state.get('text'))

if st.session_state.get('text'):
    st.subheader("그림 그리기 🎨 AI Tory가 그림을 그려줄게요 🪄")
    query = st.text_input("원하는 그림에 대해서 설명 해주세요!", placeholder="Send a message")
    # "그려줘" 버튼을 눌렀을 때 대화 내용 출력
    if query:
            conversation = ""
            conversation += f"{st.session_state['text']}"
            user_input = conversation

            gpt_prompt = [{
                "role" : "system", 
                "content" : f"""In {user_input}, you are Beatrix Potter drawing. 
                            I want to draw a children's storybook about {query}. Imagine the details.
                            """

            }]
            
            # gpt_prompt.append({
            #         "role" : "user",
            #         "content" : query
            # })

            with st.spinner("토리가 동화를 상상하고 있어요.."):
                    gpt_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-16k",
                        messages=gpt_prompt,
                        max_tokens=150
                    )

            pic_prompt = gpt_response["choices"][0]["message"]["content"]
            dalle_prompt = pic_prompt
            st.info(dalle_prompt)
            with st.spinner("토리가 동화에 대해서 그려줄게요.."):
                dallE_response = openai.Image.create(
                    prompt=dalle_prompt,
                    size= "512x512",
                    n=4
                )
                
            translator = googletrans.Translator()
            ko_dalle_prompt = translator.translate(dalle_prompt, dest='ko')

            # 이미지를 2x2 배열로 정렬
            num_images = len(dallE_response["data"])
            if num_images >= 4:
                # 2x2 배열을 만들기 위해 4개의 이미지를 2개씩 나눔
                rows = [dallE_response["data"][:2], dallE_response["data"][2:4]]
            else:
                rows = [dallE_response["data"]]  # 4개 미만의 이미지는 그대로 표시

            # 각 행에 이미지 배치
            for row_images in rows:
                columns = st.columns(len(row_images))
                for i, image_data in enumerate(row_images):
                    with columns[i]:
                        st.image(image_data["url"])

            st.success(ko_dalle_prompt.text)

