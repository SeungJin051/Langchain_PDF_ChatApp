import os
#  streamlit = Python에서 GUI 생성
import streamlit as st
import pickle # 파이썬 객체를 바이너리 파일로 저장하고 불러오는 기능
import playsound

from tempfile import NamedTemporaryFile

from gtts import gTTS

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
from dotenv import load_dotenv # OPEN_API_KEY
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain # 답변

# ----------------------

# 수직 사이드바
with st.sidebar:
    st.title('🤖 LLM AI Tory')
    st.markdown('''
    ## About
    AI Tory와 함께 놀아요! 😀
    - [AI Tory와 대화하기](https://streamlit.io/)
    - [주인공과 대화하기](https://streamlit.io/)
    - [그림 그리기](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write('Project [AI Tory](https://python.langchain.com/)')
# 수직 사이드바

# ----------------------


# ----------------------
def main():
    load_dotenv()

    st.header("AI Tory와 대화하기! 💬")    

    # upload PDF / 파일당 200MB 제한
    pdf = st.file_uploader("AI Tory에게 학습할 동화 PDF를 Upload 해주세요", type='pdf', key='pdf')

    # 업로드가 없을시 None 방지 if문
    if pdf is not None:

        query = st.text_input("AI토리에게 질문하세요!",)
        AIttsButton = st.button("🔊")
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
        chunks = text_splitter.split_text(text = text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"pdfs/{store_name}.pkl"):
            with open(f"pdfs/{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            print("해당 PDF는 저장소에 있습니다!")
        else :
            embedding = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding = embedding)
            with open(f"pdfs/{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            print("해당 PDF는 저장소에 없습니다!")

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        
        if 'past' not in st.session_state:
            st.session_state['past'] = []

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)

            prompt ="""
                [예외]
                - 저는 '{pdf.name}'라는 동화를 학습했어요. {query}은 모르겠어요..

                -----

                [학습 정보]
                - {text}

                -----

                [규칙]
                - [학습 정보]에 없는 질문인 경우 [예외]로 대답한다.
                - '{pdf.name}'를 학습했습니다.
                - 오직 [학습 정보]에서 배운 내용만 답하세요. 
                - 마치 어린 아이와 대화하는 것처럼 친절한 어조와 간단한 단어로 작성

                -----

                위 정보는 모두 {pdf.name} 동화에 관련된 내용입니다. 
                [예외], [학습 정보], [규칙]은 답변하지 않는다.
                AI Tory가 [학습정보에 대한 정보로 창의적인 답변을 합니다.
                당신은 오직 학습된 내용만 알려주며 [규칙]을 무조건 따르는 AI Tory입니다.
                질문하는 사용자에게 [학습 정보]로 학습된 내용을 기반으로 답변해야합니다. 
                [예시 문장]과 [학습 정보]를 참고하여 다음 조건을 만족하면서 [학습 정보]에 있는 정보 메시지를 창의적인 답변을 합니다.
            """

            user_question = prompt + query

            with st.spinner("토리가 생각하고 말해줄게요!"):
                llm = OpenAI(
                    temperature=0.5,
                    model_name="gpt-3.5-turbo-16k",
                    max_tokens=600,
                    top_p=1)

                chain = load_qa_chain(llm=llm, chain_type="stuff")
                response = chain.run(input_documents=docs, question=user_question)

                bot_message = response
                output = bot_message

                st.session_state.past.append(query)
                st.session_state.generated.append(output)




            with st.spinner("토리가 말해줄게요!"):
                # 만약 'generated'라는 세션 상태 변수가 존재한다면
                if st.session_state['generated']:
                # 'generated' 리스트의 마지막 요소부터 역순으로 반복
                    for i in range(len(st.session_state['generated']) - 1, -1, -1):
                        # 사용자 메시지를 표시 (세션 상태의 'past' 변수를 사용)
                        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                        # 생성된 메시지를 표시 (고유한 키 값을 설정하여 중복을 피함)
                        message(st.session_state["generated"][i], key=str(i))

                if AIttsButton:
                    # gTTS를 사용하여 텍스트를 음성으로 변환
                    tts = gTTS(text=output, lang='ko')  # 'ko'는 한국어
                    temp_file = NamedTemporaryFile(delete=False)
                    tts.save(temp_file.name)

                    # 임시 파일을 재생
                    playsound.playsound(temp_file.name)
                    # 임시 파일 삭제
                    temp_file.close()


                        
if __name__ == "__main__":
    main()
