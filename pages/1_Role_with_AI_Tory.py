import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound
import openai
import pyaudio
import wave

from side_bar import run_side_bar, set_bg_hack 

from PyPDF2 import PdfReader # PyPDF2 = streamlit의 PDF 업로드를 읽기 위해 
from tempfile import NamedTemporaryFile

from components import tory_firebase

from gtts import gTTS
from dotenv import load_dotenv # OPEN_API_KEY

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain # 답변
from langchain.text_splitter import RecursiveCharacterTextSplitter # langchain.text_splitter = PyPDF2의 텍스트를 chunks로 나눔
from langchain.embeddings.openai import OpenAIEmbeddings # openAI의 embedding = 계산하고 반환
from langchain.vectorstores import FAISS # VectorStore = FAISS, Chroma X = VectorStore에서 duckdb.DuckDBPyConnection 접근 불가능
# -------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pdf_image_path = "pages/images/tory_pdf.png" 
main_bg_ext = "pages/images/tory_back.png"

set_bg_hack(main_bg_ext)
st.write( """
        <style>
        .st-bd {
            background-color: rgba(255, 255, 255);
            border-radius: 15px; 
            padding: 30px; 
            box-shadow: 5px 5px 5px rgba(0, 0, 0, 0.2); 
        }
        </style>
        """,
    unsafe_allow_html=True)

# 사이드 바 생성
pdf, text, VectorStore = run_side_bar()
sample_rate = 44100  # 오디오 샘플 속도
duration = 6  # 녹음 시간 (초)
tab1, tab2 = st.tabs([" ", " "])

with tab1:
    if pdf is None:
        # 스트림릿 앱 헤더 설정
        st.header("AI Tory와 역할놀이 하기! 🕹️")
        st.caption('AI Tory에게 PDF를 학습시키고, 역할을 입력하고 역할놀이를 해봐요! 🫵🏻')
        st.image(pdf_image_path)

    if pdf is not None:
        st.header(f"AITory와 {pdf.name} 역할놀이 🕹️")
        st.caption('AI Tory에게 PDF를 학습시키고, 역할을 입력하고 역할놀이를 해봐요! 🫵🏻')
    if pdf is not None:
        user = ""
        ai = ""

        # 사용자 및 AI 역할 입력
        with st.expander(' ', expanded=True):
            user_input_col, ai_input_col = st.columns(2)
            user_input = user_input_col.text_input("사용자의 역할을 입력하세요 :")
            ai_input = ai_input_col.text_input("AI Tory의 역할을 입력하세요 :")

            if not user_input or not ai_input:
                st.warning("역할을 입력해주세요.")
        
        col5, col6, col7 = st.columns(3)

        if user_input and ai_input:  # 역할이 모두 입력된 경우에만 아래 컨테이너 표시

            # 스트림릿 컨테이너 생성
            with st.container():
                with st.form(key='role_form', clear_on_submit=True):
                    query = st.text_input(' ',  placeholder="Send a message", value="")
                    col5, col6, col7 = st.columns(3)

                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    with col1:
                        submit = st.form_submit_button(label="Send")
                    with col2:
                        whisper_button = st.form_submit_button("🎙️", help="마이크를 연결해주세요.")
                    with col3:
                        tts_button = st.checkbox("🔊",  value=False, help="AI토리가 말해줄게요.")
                    with col4:
                        toggle_state = st.checkbox('AI 🎨', value=True, help="AI토리가 그림을 그려줄게요.")

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
                # PDF 파일에서 텍스트 추출
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # 텍스트를 적절한 크기의 청크로 나누기
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text=text)

                # PDF 파일 이름으로 저장소 생성 또는 로드
                store_name = pdf.name[:-4]

                if os.path.exists(f"pdfs/{store_name}.pkl"):
                    with open(f"pdfs/{store_name}.pkl", "rb") as f:
                        VectorStore = pickle.load(f)
                    print("해당 PDF는 저장소에 있습니다!")
                else:
                    embedding = OpenAIEmbeddings()
                    VectorStore = FAISS.from_texts(chunks, embedding=embedding)
                    with open(f"pdfs/{store_name}.pkl", "wb") as f:
                        pickle.dump(VectorStore, f)
                    print("해당 PDF는 저장소에 없습니다!")

                # 세션 상태 변수 초기화
                if 'role_generated' not in st.session_state:
                    st.session_state['role_generated'] = []

                if 'role_past' not in st.session_state:
                    st.session_state['role_past'] = []

                if query:
                    # 유사한 문서 검색을 통해 적절한 문서 가져오기
                    docs = VectorStore.similarity_search(query=query, k=3)
                    user = user_input
                    ai = ai_input
                    print(user)

                    # AI Tory에게 전달할 질문 작성
                    prompt = f"""
                    너 = {user}
                    나 = {ai}
                    사용자의 질문 = {query}

                    ----------------------------------------------------------------

                    [예시문장]
                    - 당신은 {user}이고, 저는 {ai}입니다.
                    - 그것에 대한 정보는 모르겠습니다.
                    
                    ----------------------------------------------------------------
                    [규칙]
                    - {user}의 {query}에 {ai}의 답변을 최대 3줄로 해라.
                    - {user}의 {query}와 {ai}의 관계를 이해하고 적절한 답변을 해라.
                    - {user}와 {ai}의 대화를 진행하세요.
                    - {user}가 지정한 {ai}에 대한 질문이 아닌 경우 모른다고 대답해.
                    - 당신은 사용자가 지정한 역할인 {ai}인 것 처럼 대답하고 행동하여야 합니다.
                    ----------------------------------------------------------------
                    위 정보는 모두 {ai}에 대한 내용입니다. [예시 문장]을 참고해서 답변해, 역할놀이에 대답한 내용입니다.
                    [규칙]을 따르는 {ai}입니다. {user}에게 {ai}만의 답변을 하세요.
                    ----------------------------------------------------------------
                    """

                    user_question = prompt + query

                    # AI Tory의 응답 생성 및 저장
                    with st.spinner("토리가 생각하고있어요..."):
                        llm = OpenAI(
                            temperature=0.1,
                            model_name="gpt-3.5-turbo-16k",
                            top_p=1,
                        )
                        chain = load_qa_chain(llm=llm, chain_type="refine")
                        response = chain.run(input_documents=docs, question=user_question)

                        bot_message = response
                        output = bot_message

                        st.session_state.role_past.append(query)
                        st.session_state.role_generated.append(output)

                    if toggle_state:
                            query = ""
                            # PDF가 업로드되었다면 PDF 처리를 합니다
                            gpt_prompt = [{
                                    "role" : "system", 
                                    "content" : f"You are Beatrix Potter. Choose a character and darw a cute and lovely, kids-like picture around that character."
                            }]
                            gpt_prompt.append({
                                    "role" : "user",
                                    "content" :f"{pdf.name}, {text}"
                            })

                            with st.spinner("토리가 동화를 상상하고 있어요.."):
                                    gpt_response = openai.ChatCompletion.create(
                                        model="gpt-3.5-turbo-16k",
                                        messages=gpt_prompt,
                                        max_tokens=50
                                    )

                            pic_prompt = gpt_response["choices"][0]["message"]["content"]
                            dalle_prompt = pic_prompt

                            with st.spinner("토리가 동화에 대해서 그려줄게요.."):
                                dallE_response = openai.Image.create(
                                    prompt=dalle_prompt,
                                    size= "1024x1024",
                                    n=1
                                )
                            with col6:
                                img = st.image(dallE_response["data"][0]["url"], caption=pdf.name)
                                st.empty()
                                st.empty()
                    # 대화 기록 및 음성 출력
                    with st.spinner("토리가 말하고있어요..."):
                        if st.session_state['role_generated']:
                            for i in range(len(st.session_state['role_generated']) - 1, -1, -1):
                                message(st.session_state["role_generated"][i], key=str(i), avatar_style="thumbs", seed="Felix")
                                message(st.session_state['role_past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs", seed="Aneka")

                            if tts_button:
                                query = ""
                                tts = gTTS(text=output, lang='ko')
                                temp_file = NamedTemporaryFile(delete=False)
                                tts.save(temp_file.name)
                                playsound.playsound(temp_file.name)
                                temp_file.close()

                        tory_firebase.add_firebase_role(query, bot_message)