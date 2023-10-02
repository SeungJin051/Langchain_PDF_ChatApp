import os
import streamlit as st #  streamlit = Python에서 GUI 생성
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound
import openai
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
pdf_image_path = "pages/images/download-pdf.gif" 

# 사이드 바 생성
pdf, text, VectorStore = run_side_bar()

if pdf is None:
    # 스트림릿 앱 헤더 설정
    st.header("AI Tory와 역할놀이 하기! 🕹️")
    st.caption('AI Tory에게 PDF를 학습시키고, 역할을 입력하고 역할놀이를 해봐요! 🫵🏻')
    st.info("PDF를 업로드 해주세요...")
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

    if user_input and ai_input:  # 역할이 모두 입력된 경우에만 아래 컨테이너 표시
        st.markdown("<hr>", unsafe_allow_html=True)

        # 스트림릿 컨테이너 생성
        with st.container():
            # 사용자 질문 입력
            query = st.text_input("AI토리에게 질문하세요!")

            # 음성 듣기 및 전송 버튼 생성
            AIttsButton = st.button("🔊")
            
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
                {user} = 상대방
                {ai} = 나
                {query} = 사용자의 질문  

                ----------------------------------------------------------------

                [예시문장]
                - 저는 백설공주입니다.
                - 당신은 왕자입니다.
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

                # 대화 기록 및 음성 출력
                with st.spinner("토리가 말하고있어요..."):
                    if st.session_state['role_generated']:
                        for i in range(len(st.session_state['role_generated']) - 1, -1, -1):
                            message(st.session_state["role_generated"][i], key=str(i), avatar_style="thumbs", seed="Felix")
                            message(st.session_state['role_past'][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs", seed="Aneka")

                        if AIttsButton:
                            tts = gTTS(text=output, lang='ko')
                            temp_file = NamedTemporaryFile(delete=False)
                            tts.save(temp_file.name)
                            playsound.playsound(temp_file.name)
                            temp_file.close()
                    tory_firebase.add_firebase_role(query, bot_message)