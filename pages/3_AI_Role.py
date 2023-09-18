import os
#  streamlit = Python에서 GUI 생성
import streamlit as st
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound

from tempfile import NamedTemporaryFile

from gtts import gTTS
from dotenv import load_dotenv # OPEN_API_KEY

from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_chat import message

# PyPDF2 = streamlit의 PDF 업로드를 읽기 위해 
from PyPDF2 import PdfReader

# langchain.text_splitter = PyPDF2의 텍스트를 chunks로 나눔
from langchain.text_splitter import RecursiveCharacterTextSplitter
# openAI의 embedding = 계산하고 반환
from langchain.embeddings.openai import OpenAIEmbeddings
# VectorStore = FAISS, Chroma X = VectorStore에서 duckdb.DuckDBPyConnection 접근 불가능
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain # 답변


import side_bar
#  모듈
from components import tory_firebase
# -------

# .env 파일로부터 환경 변수 로드
load_dotenv()

# 사이드 바 생성
side_bar.run_side_bar()

# 스트림릿 앱 헤더 설정
st.header("AI Tory와 역할극 하기! 🕹️")
st.caption('AI Tory에게 PDF를 학습시키고, 역할을 입력하고 역할놀이를 해봐요! 🫵🏻')

# PDF 파일 업로드 및 사용자 질문 입력
pdf = st.file_uploader(label=' ', type='pdf', key='pdf', help='AI Tory에게 학습할 동화 PDF를 Upload 해주세요') 
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # 사용자 질문 입력
    query = st.text_input("AI토리에게 질문하세요!")

    # 음성 듣기 및 전송 버튼 생성
    _, _, _, AIttsButton_col, submit_button_col = st.columns([2, 1, 1, 1, 1])
    AIttsButton = AIttsButton_col.button("음성 듣기 🔊")
    submit_button = submit_button_col.button("전송하기")

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
        {store_name} = 학습된 동화의 제목
        {text} = 학습된 동화의 내용
        {user} = 나 (사용자)
        {ai} = 너 (답변)
        {query} = 사용자의 질문  


        [예시 문장]
            - 너는 {user}야.
            - 당신은 {user}입니다.
            - 히히히 내가 너한테 먹인 음식은 빨간 독사과였지. 그 사과는 제가 해골에 마법을 거는 마법으로 만들었던 것이고 이 사과를 먹게 되면 너는 깊은 잠에 빠지게 된단다. 하하하!!
            - 백설공주님이 세상에서 제일 아름답습니다. 그녀는 흰 눈, 붉은 입술, 검은 머리칼에 살짝 붉은 뺨을 가지고 있어요.
            - 너는 마녀가 준 독사과를 먹고 쓰러져있었어.
            - 내가 백설공주를 깨워드리겠소.

        ----------------------------------------------------------------
        [규칙]
        - {user}의 {query}와 {ai}의 관계를 이해하고 적절한 답변을 해라.
        - 나 = {user}
        - 너 = {ai}
        - {user}와 {ai}의 대화를 진행하세요.
        - {user}가 지정한 {ai}에 대한 질문이 아닌 경우 모른다고 대답해.
        - 당신은 사용자가 지정한 역할인 {ai}인 것 처럼 대답하고 행동하여야 합니다.
        ----------------------------------------------------------------
        위 정보는 모두 {ai}에 대한 내용입니다. [예시 문장]은 {text}를 기반으로 역할놀이에 대답한 내용입니다.
        [규칙]을 따르는 {ai}입니다. {user}에게 [예시 문장]을 참고하여 {ai}만의 답변을 1문장으로 답변하세요.
        ----------------------------------------------------------------
        """

        user_question = prompt + query

        # AI Tory의 응답 생성 및 저장
        with st.spinner("토리가 생각하고있어요..."):
            llm = OpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo-16k",
                top_p=1,
                max_tokens=1000
            )
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            bot_message = response
            output = bot_message

            st.session_state.role_past.append(query)
            st.session_state.role_generated.append(output)

        # 대화 기록 및 음성 출력
        with st.spinner("토리가 말하고있어요..."):
            if st.session_state['role_generated']:
                for i in range(len(st.session_state['role_generated']) - 1, -1, -1):
                    message(st.session_state['role_past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs", seed="Aneka")
                    message(st.session_state["role_generated"][i], key=str(i), avatar_style="thumbs", seed="Felix")

        if AIttsButton:
            tts = gTTS(text=output, lang='ko')
            temp_file = NamedTemporaryFile(delete=False)
            tts.save(temp_file.name)
            playsound.playsound(temp_file.name)
            temp_file.close()
        tory_firebase.add_firebase_role(query, bot_message)