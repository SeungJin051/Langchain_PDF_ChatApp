import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound
import openai
import pyaudio
import wave

from side_bar import run_side_bar

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

# .env 파일로부터 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
thumb_style_image_path = "pages/images/thumb_style.png" 
pdf_image_path = "pages/images/download-pdf.gif" 
# 사이드 바 생성
pdf, text, VectorStore = run_side_bar()
sample_rate = 44100  # 오디오 샘플 속도
duration = 6  # 녹음 시간 (초)

tab1, tab2 = st.tabs(["AI Tory와 대화하기", "AI Tory역할놀이 하기"])
with tab1:
    if pdf is None:
        # 스트림릿 앱 헤더 설정
        st.header("AI Tory와 대화하기! 💬")
        st.caption('AI Tory에게 PDF를 학습시키고, 함께 이야기하며 혁신적인 아이디어를 공유해보세요! 💡')
        st.image(pdf_image_path)

    if pdf is not None:
        st.header(f"AITory와 {pdf.name} 💬")
        st.caption('AI Tory에게 PDF를 학습시키고, 함께 이야기하며 혁신적인 아이디어를 공유해보세요! 💡')
            
    # 세션 상태 변수 초기화
    if 'chat_generated' not in st.session_state:
        st.session_state['chat_generated'] = []

    if 'chat_past' not in st.session_state:
        st.session_state['chat_past'] = []

    if 'text' not in st.session_state:
        st.session_state['text'] = []
    

    col1, col2, col3 = st.columns(3)

    if pdf is not None:
        query = st.text_input("AI토리에게 질문하세요!", placeholder="Send a message")
        st.session_state.text.append(text)

        # 가로로 정렬된 버튼
        btn_col1, btn_col2, btn_col3 = st.columns(3)

        with btn_col1:
            whisper_button = st.button("🎙️", help="마이크를 연결해주세요.")

        with btn_col2:
            tts_button = st.checkbox("🔊",  value=True, help="AI토리가 말해줄게요.")

        with btn_col3:
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

        if toggle_state is False: 
            with col2:
                img = st.image(thumb_style_image_path, caption="AITory")
                st.empty()
                st.empty()

        if query:
            # 유사한 문서 검색을 통해 적절한 문서 가져오기
            docs = VectorStore.similarity_search(query=query, k=3)

            # AI Tory에게 전달할 질문 작성
            prompt = f"""
                {pdf.name} = 학습된 동화의 제목
                {text} = 학습된 동화의 내용
                [예외]
                - 저는 {pdf.name}라는 동화를 학습했어요. 모르겠어요..

                -----

                [학습 정보]
                - {text}

                -----

                [규칙]
                - 저는 한국어로만 답변합니다.
                - {pdf.name}를 학습했습니다.
                - 오직 [학습 정보]에서 배운 내용만 답하세요. 
                - 마치 어린 아이와 대화하는 것처럼 친절한 어조와 간단한 단어로 작성

                -----

                위 정보는 모두 {pdf.name} 동화에 관련된 내용입니다. 
                [예외], [학습 정보], [규칙]은 답변하지 않는다.
                AI Tory가 [학습정보]에 대한 정보로 창의적인 답변을 합니다.
                당신은 오직 학습된 내용만 알려주며 [규칙]을 무조건 따르는 AI Tory입니다.
                질문하는 사용자에게 [학습 정보]로 학습된 내용을 기반으로 답변해야합니다. 
                [예시 문장]과 [학습 정보]를 참고하여 다음 조건을 만족하면서 [학습 정보]에 있는 정보 메시지를 창의적인 답변을 합니다.
            """

            user_question = prompt + query

            # AI Tory의 응답 생성 및 저장
            with st.spinner("토리가 생각하고있어요..."):
                llm = OpenAI(
                    temperature=0.1,
                    model_name="gpt-3.5-turbo-16k",
                    top_p=1,
                    max_tokens=1000,
                )

                chain = load_qa_chain(llm=llm, chain_type="stuff") 
                response = chain.run(input_documents=docs, question=user_question)

                bot_message = response
                output = bot_message

                st.session_state.chat_past.append(query)
                st.session_state.chat_generated.append(output)

            if toggle_state:
                    # PDF가 업로드되었다면 PDF 처리를 합니다
                    gpt_prompt = [{
                        "role" : "system", 
                        "content" : f"""
                                    [Information] = {text}
                                    You are Beatrix Potter.
                                    You are draw a children's comic about {query}.
                                    Imagine the details.
                                    You Must be drawing for children's.
                                    """
                    }]
                    # gpt_prompt = [{
                    #         "role" : "system", 
                    #         "content" : f"You are a great painter that children like. Choose one character and organize the contents so that a fairy tale book can be created around that character. It is cute and draws pictures that children will like."
                    # }]
                    # gpt_prompt.append({
                    #         "role" : "user",
                    #         "content" :f"{pdf.name}, {text}"
                    # })

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
                    with col2:
                        img = st.image(dallE_response["data"][0]["url"], caption=pdf.name)
                        st.empty()
                        st.empty()
        

            # 대화 기록 및 음성 출력, tts 토글버튼
            with st.spinner("토리가 말하고있어요..."):
                if st.session_state['chat_generated']:
                    for i in range(len(st.session_state['chat_generated']) - 1, -1, -1):
                        message(st.session_state['chat_past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs", seed="Aneka")
                        message(st.session_state["chat_generated"][i], key=str(i), avatar_style="thumbs", seed="Felix")
                    
                    if tts_button:
                        tts = gTTS(text=output, lang='ko')
                        temp_file = NamedTemporaryFile(delete=False)
                        tts.save(temp_file.name)
                        playsound.playsound(temp_file.name)
                        temp_file.close()

            tory_firebase.add_firebase_chat(query, bot_message)
with tab2:
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
        with st.expander('역할을 정해주세요!', expanded=True):
            user_input_col, ai_input_col = st.columns(2)
            user_input = user_input_col.text_input("사용자의 역할을 입력하세요:")
            ai_input = ai_input_col.text_input("AI Tory의 역할을 입력하세요:")

            if not user_input or not ai_input:
                st.warning("역할을 입력해주세요.")
                
        col1, col2, col3 = st.columns(3)

        if user_input and ai_input:  # 역할이 모두 입력된 경우에만 아래 컨테이너 표시
            st.markdown("<hr>", unsafe_allow_html=True)

            # 스트림릿 컨테이너 생성
            with st.container():
                # 사용자 질문 입력
                query = st.text_input("AI토리에게 질문하세요!", placeholder="Send a message", key="unique_key_for_text_input")

                # 가로로 정렬된 버튼
                btn_col1, btn_col2, btn_col3 = st.columns(3)

                with btn_col1:
                    whisper_button = st.button("🎙️", help="마이크를 연결해주세요.", key="unique_key_for_whisper_button")

                with btn_col2:
                    tts_button = st.checkbox("🔊",  value=True, help="AI토리가 말해줄게요.", key="unique_key_for_tts_button")

                with btn_col3:
                    toggle_state = st.checkbox('AI 🎨', value=True, help="AI토리가 그림을 그려줄게요.", key="unique_key_for_toggle_state")

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
                if toggle_state is False: 
                    with col2:
                        st.empty()
                        st.empty()
                        st.empty()

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
                                "content" : f"""
                                            [Information] = {user_input}.
                                            You are Beatrix Potter.
                                            You are draw a children's comic draw pic about {query}.
                                            Imagine the details, No need text.
                                            You Must be drawing for children's.
                                            If say something that's not in the [Information], just draw your own picture of {query}
                                            """
                            }]    

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
                            with col2:
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