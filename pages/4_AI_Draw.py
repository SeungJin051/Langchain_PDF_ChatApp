import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import side_bar
import openai

from dotenv import load_dotenv # OPEN_API_KEY

# -------

# .env 파일로부터 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 사이드 바 생성
side_bar.run_side_bar()

# 스트림릿 앱 헤더 설정
st.header("AI Tory의 그림 그리기 🎨")
st.caption('토리와의 역할놀이에서 저장된 정보를 기반으로 그림을 그려줄게요 🔥')

st.markdown("""
<style>
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# 버튼을 생성합니다.
DrawButton = st.button("그림 그리기 🎨 AI Tory와 함께 그림을 만들어보세요 🪄")
# "그려줘" 버튼을 눌렀을 때 대화 내용 출력
if DrawButton:
    if 'role_generated' not in st.session_state or not st.session_state['role_generated']:
        st.warning("역할놀이를 먼저 해봐요!")
    else:
        conversation = ""
        for i in range(len(st.session_state['role_generated']) - 1, -1, -1):
            conversation += f"{st.session_state['role_generated'][i]}\n\n"
        st.write(conversation)
        user_input = conversation


        gpt_prompt = [{
            "role" : "system", 
            "content" :  f"""
                            Understand Korean by translating it into English
                            Find a person in {user_input}
                            Write explanatory, focusing on nouns and visually descriptive phrases.
                            Use terms from relevant fields such as art techniques, art media, and artist names to describe styles.
                            When describing a genre, use short sections separated by commas and suggest a genre by combining compatible artists and styles.
                            Example:
                            Expanding ideas
                            Original prompt: Fairy Tale creating a cute anthropomorphic fox character dressed in colorful costumes and holding balloons
                            Be the main character.
                        
                         """
        }]
                            # Summarize in one line.
        
        gpt_prompt.append({
            "role" : "user",
            "content" : user_input
        })

        with st.spinner("토리가 어떤 그림을 그릴지 생각하고 있어요.."):
            gpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=gpt_prompt
            )

        prompt = gpt_response["choices"][0]["message"]["content"]

        dalle_prompt = gpt_response

        st.write("dalle_prompt = " + prompt)

        with st.spinner("토리가 열심히 그림을 그리고 있어요.."):
            dallE_response = openai.Image.create(
                prompt=prompt,
                size="1024x1024",
            )

        st.image(dallE_response["data"][0]["url"], caption=dalle_prompt)
