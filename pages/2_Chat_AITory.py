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
from components import tory_firebase

# .env 파일로부터 환경 변수 로드
load_dotenv()

# 사이드 바 생성
side_bar.run_side_bar()

# 스트림릿 앱 헤더 설정
st.header("AI Tory와 대화하기! 💬")

# PDF 파일 업로드 및 사용자 질문 입력
pdf = st.file_uploader("AI Tory에게 학습할 동화 PDF를 Upload 해주세요", type='pdf', key='pdf')

if pdf is not None:
    query = st.text_input("AI토리에게 질문하세요!")

    AIttsButton = st.button("🔊")
    pdf_reader = PdfReader(pdf)
    text = ""

    # 업로드한 PDF에서 텍스트 추출
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
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

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
            - 역할놀이
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
                max_tokens=1000
            )

            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            bot_message = response
            output = bot_message


            st.session_state.past.append(query)
            st.session_state.generated.append(output)

        # 대화 기록 및 음성 출력
        with st.spinner("토리가 말하고있어요..."):
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated']) - 1, -1, -1):
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["generated"][i], key=str(i))

                if AIttsButton:
                    tts = gTTS(text=output, lang='ko')
                    temp_file = NamedTemporaryFile(delete=False)
                    tts.save(temp_file.name)
                    playsound.playsound(temp_file.name)
                    temp_file.close()
        tory_firebase.add_firebase(query, bot_message)