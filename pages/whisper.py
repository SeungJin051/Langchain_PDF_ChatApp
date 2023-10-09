import streamlit as st
import pyaudio
import wave
import openai

# Streamlit 앱 설정
st.title("마이크 녹음 예제")

# 오디오 녹음 관련 설정
sample_rate = 44100  # 오디오 샘플 속도
duration = 5  # 녹음 시간 (초)
# 버튼을 사용하여 마이크 녹음 시작
if st.button("마이크 활성화 및 녹음 시작"):
    st.write("Recording...")

    # PyAudio를 사용하여 오디오 스트림 열기
    audio_data = []
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=sample_rate, input=True, frames_per_buffer=1024)

    # 오디오 데이터 녹음
    for i in range(0, int(sample_rate / 1024 * duration)):
        audio_chunk = stream.read(1024)
        audio_data.append(audio_chunk)

    # 녹음 중지
    stream.stop_stream()
    stream.close()
    p.terminate()

    st.write("Recording done!")

    # 녹음된 오디오를 파일로 저장 (옵션)
    audio_file = "recorded_audio.wav"
    with wave.open(audio_file, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(audio_data))

    st.audio(audio_file, format="audio/wav")  # 녹음된 오디오를 스트림리트 앱에 출력

    # 수정된 부분: 녹음된 오디오 파일을 읽기 모드로 열기
    with open("recorded_audio.wav", "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        ko_response = transcript["text"].encode('utf-16').decode('utf-16')
        query = st.text_input("hello", value=ko_response)

if 'something' not in st.session_state:
    st.session_state.something = ''

def submit():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

st.text_input('Something', key='widget', on_change=submit)

st.write(f'Last submission: {st.session_state.something}')