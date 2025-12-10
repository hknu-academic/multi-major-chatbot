import streamlit as st
import pandas as pd
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="다전공제도 안내 챗봇",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"  # 모바일에서 사이드바 자동 접힘
)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = {}

# === 데이터 로드 함수 ===
@st.cache_data
def load_programs():
    """제도 정보 로드"""
    try:
        df = pd.read_excel('data/programs.xlsx')
        programs = {}
        for _, row in df.iterrows():
            programs[row['제도명']] = {
                'description': row['설명'],
                'credits': row['이수학점'],
                'qualification': row['신청자격'],
                'degree': row['학위표기'],
                'difficulty': '★' * int(row['난이도']) + '☆' * (5 - int(row['난이도'])),
                'features': row['특징'].split(',') if pd.notna(row.get('특징')) else []
            }
        return programs
    except FileNotFoundError:
        st.warning("⚠️ data/programs.xlsx 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        return get_sample_programs()
    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {e}")
        return get_sample_programs()

@st.cache_data
def load_faq():
    """FAQ 로드"""
    try:
        df = pd.read_excel('data/faq.xlsx')
        return df.to_dict('records')
    except FileNotFoundError:
        st.warning("⚠️ data/faq.xlsx 파일을 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        return get_sample_faq()
    except Exception as e:
        st.error(f"❌ FAQ 로드 오류: {e}")
        return get_sample_faq()

@st.cache_data
def load_curriculum_mapping():
    """이수체계도 이미지 매핑 로드"""
    try:
        df = pd.read_excel('data/curriculum_mapping.xlsx')
        return df
    except FileNotFoundError:
        st.warning("⚠️ data/curriculum_mapping.xlsx 파일을 찾을 수 없습니다.")
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])
    except Exception as e:
        st.error(f"❌ 매핑 데이터 로드 오류: {e}")
        return pd.DataFrame(columns=['전공명', '제도유형', '파일명'])

@st.cache_data
def load_courses():
    """과목 정보 로드"""
    try:
        df = pd.read_excel('data/courses.xlsx')
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=['전공명', '제도유형', '과목코드', '과목명', '학점', '필수여부'])
    except Exception as e:
        st.error(f"❌ 과목 데이터 로드 오류: {e}")
        return pd.DataFrame(columns=['전공명', '제도유형', '과목코드', '과목명', '학점', '필수여부'])

# 샘플 데이터 (엑셀 파일이 없을 때)
def get_sample_programs():
    return {
        "복수전공": {
            "description": "주전공 외에 다른 전공을 추가로 이수하여 2개의 학위를 취득하는 제도",
            "credits": "36학점 이상",
            "qualification": "2학년 이상, 평점 2.0 이상",
            "degree": "2개 학위 수여",
            "difficulty": "★★★★☆",
            "features": ["졸업 시 2개 학위 취득", "취업 시 경쟁력 강화", "학점 부담 높음"]
        },
        "부전공": {
            "description": "주전공 외에 다른 전공의 기초과목을 이수하는 제도",
            "credits": "21학점 이상",
            "qualification": "2학년 이상",
            "degree": "주전공 학위 (부전공 표기)",
            "difficulty": "★★☆☆☆",
            "features": ["학점 부담 적음", "학위증에 부전공 표기"]
        }
    }

def get_sample_faq():
    return [
        {
            "카테고리": "일반",
            "질문": "복수전공과 부전공의 차이는?",
            "답변": "복수전공은 36학점 이상을 이수하여 2개의 학위를 받지만, 부전공은 21학점 이수로 주전공 학위만 받습니다."
        }
    ]

# 데이터 로드
PROGRAM_INFO = load_programs()
FAQ_DATA = load_faq()
CURRICULUM_MAPPING = load_curriculum_mapping()
COURSES_DATA = load_courses()

# === 유사도 기반 검색 함수 ===
@st.cache_resource
def create_faq_vectorizer():
    """FAQ 질문들을 벡터화"""
    questions = [faq['질문'] for faq in FAQ_DATA]
    vectorizer = TfidfVectorizer()
    
    if questions:
        vectors = vectorizer.fit_transform(questions)
        return vectorizer, vectors, questions
    return None, None, []

def find_similar_faq(user_input, threshold=0.5):
    """유사한 FAQ 찾기 (임계값 상향)"""
    vectorizer, faq_vectors, questions = create_faq_vectorizer()
    
    if vectorizer is None or not questions:
        return None
    
    # 사용자 입력 벡터화
    user_vector = vectorizer.transform([user_input])
    
    # 코사인 유사도 계산
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    # 가장 유사한 FAQ 찾기
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]
    
    # 임계값 이상이면 해당 FAQ 반환
    if max_similarity >= threshold:
        return FAQ_DATA[max_similarity_idx], max_similarity
    
    return None

def get_top_similar_faqs(user_input, top_n=3):
    """가장 유사한 FAQ 여러 개 반환"""
    vectorizer, faq_vectors, questions = create_faq_vectorizer()
    
    if vectorizer is None or not questions:
        return []
    
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    # 상위 N개 인덱스
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # 최소 10% 유사도
            results.append({
                'faq': FAQ_DATA[idx],
                'similarity': similarities[idx]
            })
    
    return results

def find_similar_program(user_input):
    """제도명 유사도 검색"""
    program_names = list(PROGRAM_INFO.keys())
    
    # 정확히 일치하는 경우
    for program in program_names:
        if program in user_input:
            return program
    
    # 부분 일치 검색
    for program in program_names:
        if any(word in user_input for word in program.split()):
            return program
    
    return None

# === 이미지 표시 함수 ===
def display_curriculum_image(major, program_type):
    """이수체계도 이미지 표시"""
    # 매핑 데이터에서 파일명 찾기
    result = CURRICULUM_MAPPING[
        (CURRICULUM_MAPPING['전공명'] == major) & 
        (CURRICULUM_MAPPING['제도유형'] == program_type)
    ]
    
    if not result.empty:
        filename = result.iloc[0]['파일명']
        image_path = f"images/curriculum/{filename}"
        
        # 이미지 파일 존재 확인
        if os.path.exists(image_path):
            st.image(image_path, caption=f"{major} {program_type} 이수체계도", use_container_width=True)
            return True
        else:
            st.warning(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")
            return False
    else:
        st.info(f"💡 {major} {program_type}의 이수체계도가 준비 중입니다.")
        return False

# === 과목 표시 함수 ===
def display_courses(major, program_type):
    """과목 정보 표시"""
    courses = COURSES_DATA[
        (COURSES_DATA['전공명'] == major) & 
        (COURSES_DATA['제도유형'] == program_type)
    ]
    
    if not courses.empty:
        st.subheader("📚 이수 과목")
        
        # 필수/선택 구분
        required = courses[courses['필수여부'] == '필수']
        elective = courses[courses['필수여부'] == '선택']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not required.empty:
                st.write("**필수 과목**")
                st.dataframe(
                    required[['과목코드', '과목명', '학점']], 
                    hide_index=True,
                    use_container_width=True
                )
        
        with col2:
            if not elective.empty:
                st.write("**선택 과목**")
                st.dataframe(
                    elective[['과목코드', '과목명', '학점']], 
                    hide_index=True,
                    use_container_width=True
                )
        
        # 총 학점 계산
        total_credits = courses['학점'].sum()
        st.info(f"📊 총 개설 학점: {total_credits}학점")
        
        return True
    else:
        return False

# === 비교표 생성 ===
def create_comparison_table():
    data = {
        "제도": list(PROGRAM_INFO.keys()),
        "이수학점": [info["credits"] for info in PROGRAM_INFO.values()],
        "학위": [info["degree"] for info in PROGRAM_INFO.values()],
        "난이도": [info["difficulty"] for info in PROGRAM_INFO.values()],
        "신청자격": [info["qualification"] for info in PROGRAM_INFO.values()]
    }
    return pd.DataFrame(data)

# === 챗봇 응답 생성 (유사도 기반) ===
def generate_response(user_input):
    user_input_lower = user_input.lower()
    
    # 인사
    if any(word in user_input_lower for word in ["안녕", "하이", "헬로", "hello", "hi"]):
        return "안녕하세요! 👋 다전공제도 안내 챗봇입니다. 복수전공, 부전공, 연계전공, 융합전공, 융합부전공, 마이크로디그리에 대해 궁금하신 점을 물어보세요!", None
    
    # 1. FAQ 유사도 검색 (우선순위 높음)
    similar_faq = find_similar_faq(user_input)
    if similar_faq:
        faq, similarity = similar_faq
        response = f"**Q. {faq['질문']}**\n\nA. {faq['답변']}\n\n"
        response += f"_💡 답변 신뢰도: {similarity*100:.0f}%_"
        return response, "faq"
    
    # 2. 제도별 정보 검색
    program = find_similar_program(user_input)
    if program:
        info = PROGRAM_INFO[program]
        response = f"**{program}** 📚\n\n"
        response += f"**설명:** {info['description']}\n\n"
        response += f"**이수학점:** {info['credits']}\n"
        response += f"**신청자격:** {info['qualification']}\n"
        response += f"**학위:** {info['degree']}\n"
        response += f"**난이도:** {info['difficulty']}\n\n"
        if info['features']:
            response += f"**특징:**\n"
            for feature in info['features']:
                response += f"- {feature}\n"
        return response, "program"
    
    # 3. 비교 질문
    if any(word in user_input_lower for word in ["비교", "차이", "다른점", "다르", "vs", "versus"]):
        return "각 제도의 비교는 왼쪽 사이드바의 '📊 제도 비교표'에서 확인하실 수 있습니다!", "comparison"
    
    # 4. 학점 관련
    if any(word in user_input_lower for word in ["학점", "몇학점", "학점수"]):
        response = "**제도별 이수 학점** 📖\n\n"
        for program, info in PROGRAM_INFO.items():
            response += f"• {program}: {info['credits']}\n"
        return response, "credits"
    
    # 5. 신청 관련
    if any(word in user_input_lower for word in ["신청", "지원", "언제", "기간", "시기"]):
        response = "**신청 관련 정보** 📝\n\n"
        response += "대부분의 제도는 매 학기 초(2월, 8월)에 신청합니다.\n\n"
        response += "자세한 내용은 '❓ FAQ' 메뉴를 확인하시거나, 학사공지를 참고해주세요!"
        return response, "application"
    
    # 6. 매칭 실패 - 유사 FAQ 제안
    similar_faqs = get_top_similar_faqs(user_input, top_n=3)
    
    if similar_faqs:
        response = "정확히 일치하는 답변을 찾지 못했습니다. 😅\n\n"
        response += "**혹시 다음 질문 중 하나를 찾으셨나요?**\n\n"
        for i, item in enumerate(similar_faqs, 1):
            faq = item['faq']
            similarity = item['similarity']
            response += f"{i}. {faq['질문']} _(유사도: {similarity*100:.0f}%)_\n"
        response += "\n💡 정확한 질문을 다시 입력해주시면 더 정확한 답변을 드릴 수 있습니다!"
        return response, "suggestion"
    
    # 7. 완전 매칭 실패
    response = "죄송합니다. 질문을 정확히 이해하지 못했습니다. 😅\n\n"
    response += "**다음과 같이 질문해보세요:**\n"
    response += "- '복수전공이 뭐야?'\n"
    response += "- '부전공과 복수전공 차이는?'\n"
    response += "- '신청은 언제 해?'\n"
    response += "- '마이크로디그리 학점은?'\n\n"
    response += "또는 왼쪽 사이드바의 **빠른 질문** 버튼이나 다른 메뉴를 이용해주세요!"
    return response, "no_match"

# === 사이드바 ===
with st.sidebar:
    st.title("🎓 다전공제도 안내")
    
    menu = st.radio(
        "메뉴 선택",
        ["💬 챗봇", "📊 제도 비교표", "❓ FAQ", "📚 전체 제도 보기", "🔍 과목 검색"]
    )
    
    st.divider()
    
    st.subheader("빠른 질문")
    quick_questions = [
        "복수전공이 뭐야?",
        "부전공 학점은?",
        "신청은 언제 해?",
        "제도 비교해줘"
    ]
    
    for q in quick_questions:
        if st.button(q, use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": q})
            response, response_type = generate_response(q)
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": response,
                "response_type": response_type
            })
            st.rerun()
    
    st.divider()
    st.caption(f"📁 로드된 제도: {len(PROGRAM_INFO)}개")
    st.caption(f"📁 로드된 FAQ: {len(FAQ_DATA)}개")
    st.caption(f"💬 피드백 수: {len(st.session_state.feedback_data)}개")

# === 메인 콘텐츠 ===
st.title("🎓 다전공제도 안내 챗봇")

if menu == "💬 챗봇":
    st.write("궁금한 점을 자유롭게 물어보세요! 😊")
    
    # 채팅 히스토리 표시
    for idx, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.write(chat["content"])
                
                # 피드백 버튼 (FAQ나 프로그램 답변에만 표시)
                if chat.get("response_type") in ["faq", "program", "comparison", "credits", "application"]:
                    feedback_key = f"feedback_{idx}"
                    
                    # 아직 피드백을 주지 않은 경우에만 버튼 표시
                    if feedback_key not in st.session_state.show_feedback:
                        col1, col2, col3 = st.columns([1, 1, 8])
                        
                        with col1:
                            if st.button("👍 도움됨", key=f"helpful_{idx}"):
                                st.session_state.feedback_data.append({
                                    "question": st.session_state.chat_history[idx-1]["content"],
                                    "answer": chat["content"],
                                    "feedback": "helpful",
                                    "timestamp": datetime.now()
                                })
                                st.session_state.show_feedback[feedback_key] = "helpful"
                                st.rerun()
                        
                        with col2:
                            if st.button("👎 아님", key=f"not_helpful_{idx}"):
                                st.session_state.feedback_data.append({
                                    "question": st.session_state.chat_history[idx-1]["content"],
                                    "answer": chat["content"],
                                    "feedback": "not_helpful",
                                    "timestamp": datetime.now()
                                })
                                st.session_state.show_feedback[feedback_key] = "not_helpful"
                                st.rerun()
                    
                    # 피드백을 준 경우 감사 메시지
                    elif st.session_state.show_feedback[feedback_key] == "helpful":
                        st.success("✅ 피드백 감사합니다!")
                    elif st.session_state.show_feedback[feedback_key] == "not_helpful":
                        st.info("📝 피드백 감사합니다. 더 나은 답변을 위해 노력하겠습니다!")
    
    # 사용자 입력
    user_input = st.chat_input("메시지를 입력하세요...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response, response_type = generate_response(user_input)
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": response,
            "response_type": response_type
        })
        st.rerun()

elif menu == "📊 제도 비교표":
    st.header("제도 비교표")
    st.write("다전공 제도를 한눈에 비교해보세요!")
    
    df = create_comparison_table()
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    st.subheader("추천 가이드")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**💪 학업 부담 감당 가능** → 복수전공")
        st.success("**⚖️ 균형잡힌 선택** → 부전공")
    
    with col2:
        st.warning("**🚀 새로운 도전** → 융합전공")
        st.info("**🎯 특정 역량 집중** → 마이크로디그리")

elif menu == "❓ FAQ":
    st.header("자주 묻는 질문 (FAQ)")
    
    categories = list(set([faq["카테고리"] for faq in FAQ_DATA]))
    selected_category = st.selectbox("카테고리 선택", ["전체"] + categories)
    
    filtered_faqs = FAQ_DATA if selected_category == "전체" else [faq for faq in FAQ_DATA if faq["카테고리"] == selected_category]
    
    for i, faq in enumerate(filtered_faqs):
        with st.expander(f"Q. {faq['질문']}"):
            st.write(f"**A.** {faq['답변']}")
            st.caption(f"카테고리: {faq['카테고리']}")

elif menu == "📚 전체 제도 보기":
    st.header("전체 제도 상세 정보")
    
    # 제도 선택
    selected_program = st.selectbox("제도 선택", list(PROGRAM_INFO.keys()))
    
    info = PROGRAM_INFO[selected_program]
    
    # 제도 정보 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{selected_program}")
        st.write(f"**📝 설명:** {info['description']}")
        st.write(f"**📖 이수학점:** {info['credits']}")
        st.write(f"**✅ 신청자격:** {info['qualification']}")
        st.write(f"**🎓 학위:** {info['degree']}")
        
        if info['features']:
            st.write("**✨ 특징:**")
            for feature in info['features']:
                st.write(f"- {feature}")
    
    with col2:
        st.metric("난이도", info['difficulty'])
    
    st.divider()
    
    # 전공 선택 (이수체계도/과목 보기용)
    if not CURRICULUM_MAPPING.empty:
        available_majors = CURRICULUM_MAPPING[
            CURRICULUM_MAPPING['제도유형'] == selected_program
        ]['전공명'].unique().tolist()
        
        if available_majors:
            selected_major = st.selectbox("전공 선택", available_majors)
            
            # 이수체계도 표시
            st.subheader("📋 이수체계도")
            display_curriculum_image(selected_major, selected_program)
            
            # 과목 정보 표시
            if not COURSES_DATA.empty:
                display_courses(selected_major, selected_program)

elif menu == "🔍 과목 검색":
    st.header("과목 검색")
    
    if not COURSES_DATA.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            search_major = st.selectbox(
                "전공 선택",
                ["전체"] + COURSES_DATA['전공명'].unique().tolist()
            )
        
        with col2:
            search_program = st.selectbox(
                "제도 선택",
                ["전체"] + COURSES_DATA['제도유형'].unique().tolist()
            )
        
        # 필터링
        filtered = COURSES_DATA.copy()
        if search_major != "전체":
            filtered = filtered[filtered['전공명'] == search_major]
        if search_program != "전체":
            filtered = filtered[filtered['제도유형'] == search_program]
        
        # 검색어
        search_keyword = st.text_input("🔍 과목명 검색")
        if search_keyword:
            filtered = filtered[filtered['과목명'].str.contains(search_keyword, na=False)]
        
        # 결과 표시
        st.write(f"검색 결과: {len(filtered)}개")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
    else:
        st.info("💡 과목 데이터가 없습니다. data/courses.xlsx 파일을 추가해주세요.")



# === 푸터 ===
st.divider()

# 피드백 통계 (관리자용)
if st.session_state.feedback_data:
    with st.expander("📊 피드백 통계 보기 (관리자용)"):
        feedback_df = pd.DataFrame(st.session_state.feedback_data)
        
        col1, col2 = st.columns(2)
        with col1:
            helpful_count = len(feedback_df[feedback_df['feedback'] == 'helpful'])
            st.metric("👍 도움됨", helpful_count)
        
        with col2:
            not_helpful_count = len(feedback_df[feedback_df['feedback'] == 'not_helpful'])
            st.metric("👎 아님", not_helpful_count)
        
        st.write("**최근 피드백**")
        st.dataframe(
            feedback_df[['question', 'feedback', 'timestamp']].tail(10),
            use_container_width=True,
            hide_index=True
        )

st.caption("💡 더 자세한 정보는 학사지원팀 또는 학과 사무실에 문의하세요.")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y년 %m월 %d일')}")
