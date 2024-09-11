import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import glob
import time
total_start_time = time.time();  start_time = time.time()

# API KEY 정보로드
load_dotenv()

st.title('나의 GPT')

# 대화기록을 저장하기 위한 용도로 생성
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.sidebar:    
    clear_btn = st.button('대화 초기화')    
    prompt_files = glob.glob('prompts/*.yaml')
    selected_prompt = st.selectbox('프롬프트를 선택해 주세요', prompt_files, index=0)
    task_input = st.text_input('TASK 입력', '')

# 초기화 벼튼을 누르면..
if clear_btn:
    st.session_state['messages'] = []

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state['messages']:
        st.chat_message(chat_message.role).write(chat_message.content)

# 새로운 메시지 추가
def add_message(role, message):
    st.session_state['messages'].append(ChatMessage(role=role, content=message))

# 체인 생성
def create_chain(prompt_filepath, task=''):
    prompt = load_prompt(prompt_filepath, encoding='utf-8')
    if task:        
        prompt = prompt.partial(task=task)
    # prompt | llm | output_parser
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ('system', '당신은 친절한 AI 챗봇입니다. 다음의 질문에 간결하게 답변해 주세요'),
    #         ('user', '#Question:\n{question}'),
    #     ]
    # )

    prompt = load_prompt(prompt_filepath, encoding='utf-8')

    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    # 출력 파서
    output_parser = StrOutputParser()
    # 체인 생성
    
    # GPT
    chain = prompt | llm | output_parser

    return chain

# 이전 대화 기록 출력
print_messages()


# 사용자 입력
user_input = st.chat_input("질문을 입력하세요")
if user_input:
    st.chat_message('user').write(user_input)
    chain = create_chain(selected_prompt, task=task_input)
    response = chain.stream({'question': user_input})
    with st.chat_message('assistant'):
        container = st.empty()
        ai_answer = ''
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    add_message('user', user_input)
    add_message('assistant', ai_answer)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
st.sidebar.write('---')
st.sidebar.write(f"#### :orange[총 검색 시간 : {time.time() - start_time:.2f} 초]")