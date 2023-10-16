import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import openai
import googletrans
import playsound
import pyaudio
import wave

from side_bar import run_side_tap_draw
from dotenv import load_dotenv # OPEN_API_KEY

# -------

# .env 파일로부터 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 사이드 바 생성
run_side_tap_draw()
sample_rate = 44100  # 오디오 샘플 속도
duration = 6  # 녹음 시간 (초)

# 스트림릿 앱 헤더 설정
st.header("그림 그리기 🎨 AI Tory가 그림을 그려줄게요 🪄")
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
    query = st.text_input("원하는 그림에 대해서 설명 해주세요!", placeholder="Send a message")
    whisper_button = st.button("🎙️", help="마이크를 연결해주세요.", key="unique_key_for_whisper_button")

    if whisper_button:
                    with st.spinner("말해주세요! 토리가 듣고있어요..."):
                        # PyAudio를 사용하여 오디오 스트림 열기
                        audio_data = []
                        p = pyaudio.PyAudio()
                        stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

                        # 오디오 데이터 녹음
                        for i in range(0, int(sample_rate / 1024 * duration)):
                            audio_chunk = stream.read(1024)
                            audio_data.append(audio_chunk)

                    with st.spinner("토리가 다 들었어요..."):
                        # 녹음 중지
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                        # 녹음된 오디오를 파일로 저장 (옵션)
                        audio_file = "recorded_audio.wav"
                        with wave.open(audio_file, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                            wf.setframerate(sample_rate)
                            wf.writeframes(b"".join(audio_data))

                        # 수정된 부분: 녹음된 오디오 파일을 읽기 모드로 열기
                        with open("recorded_audio.wav", "rb") as audio_file:
                            transcript = openai.Audio.transcribe("whisper-1", audio_file)
                            ko_response = transcript["text"].encode('utf-16').decode('utf-16')
                            query = ko_response
    if query:
            conversation = ""
            conversation += f"{st.session_state['text']}"
            user_input = conversation

            gpt_prompt = [{
                "role" : "system", 
                "content" : f"""
                            [Information] = {user_input}.
                            You are Beatrix Potter.
                            You are draw a children's comic draw pic about {query}.
                            Imagine the details, No need text.
                            You Must be drawing for children's.
                            If say something that's not in the [Information], just draw your own picture of {query}
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

