import streamlit as st
import pandas as pd
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import uuid
import re
from google import genai

# === [AI 설정] Gemini API 연결 ===
GEMINI_API_KEY = "AIzaSyD_4GxoAS9nL-YaOJ-Fb2ZYNhRx7y_uUAA"  # 제공해주신 키 적용
client = genai.Client(api_key=GEMINI_API_KEY)  # <--- Client 객체 생성 방식으로 변경

# === 페이지 설정 (가장 먼저 실행되어야 함) ===
st.set_page_config(
    page_title="한경국립대 다전공 안내 챗봇",
    page_icon="🎓",
    layout="wide",
)

# === 자동 스크롤 함수 (마지막 말풍선 추적 방식 + Focus) ===
def scroll_to_bottom():
    # 매번 새로운 ID로 강제 실행 유도
    unique_id = str(uuid.uuid4())
    
    js = f"""
    <script>
        // Random ID to force update: {unique_id}
        
        function scrollIntoView() {{
            // 1. 말풍선 요소들을 다 찾습니다.
            var messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            
            if (messages.length > 0) {{
                // 2. 가장 마지막 말풍선을 가져옵니다.
                var lastMessage = messages[messages.length - 1];
                
                // 3. 그 말풍선이 보이도록 화면을 부드럽게 내립니다.
                lastMessage.scrollIntoView({{behavior: "smooth", block: "end"}});
            }} else {{
                // 말풍선을 못 찾으면 기존 방식으로 컨테이너 스크롤 시도
                var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                if (container) container.scrollTop = container.scrollHeight;
            }}
        }}

        // 화면 렌더링 시간을 고려해 조금 넉넉히 기다렸다가 실행
        setTimeout(scrollIntoView, 300);
        setTimeout(scrollIntoView, 500);
    </script>
    """
    st.components.v1.html(js, height=0)

# 세션 상태 초기화
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'scroll_to_bottom' not in st.session_state:
    st.session_state.scroll_to_bottom = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'show_feedback' not in st.session_state:
    st.session_state.show_feedback = {}
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False
if "scroll_count" not in st.session_state:
    st.session_state.scroll_count = 0
if 'show_calculator' not in st.session_state:
    st.session_state.show_calculator = False


# 관리자 비밀번호 (환경변수 우선, 없으면 기본값)
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin1234")

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
                'credits_general': row['이수학점(교양)'] if pd.notna(row.get('이수학점(교양)')) else '-',
                'credits_primary': row['원전공 이수학점'] if pd.notna(row.get('원전공 이수학점')) else '-',
                'credits_multi': row['다전공 이수학점'] if pd.notna(row.get('다전공 이수학점')) else '-',
                'graduation_certification': row['졸업인증'] if pd.notna(row.get('졸업인증')) else '-',
                'graduation_exam': row['졸업시험'] if pd.notna(row.get('졸업시험')) else '-',
                'qualification': row['신청자격'],
                'degree': row['학위표기'],
                'difficulty': '★' * int(row['난이도']) + '☆' * (5 - int(row['난이도'])),
                'features': row['특징'].split(',') if pd.notna(row.get('특징')) else [],
                'notes': row['기타'] if pd.notna(row.get('기타')) else ''
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
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])
    except Exception as e:
        st.error(f"❌ 과목 데이터 로드 오류: {e}")
        return pd.DataFrame(columns=['전공명', '제도유형', '학년', '학기', '이수구분', '과목명', '학점'])

@st.cache_data
def load_keywords():
    """키워드 매핑 로드"""
    try:
        df = pd.read_excel('data/keywords.xlsx')
        return df.to_dict('records')
    except FileNotFoundError:
        st.warning("⚠️ data/keywords.xlsx 파일을 찾을 수 없습니다. 기본 키워드를 사용합니다.")
        return get_default_keywords()
    except Exception as e:
        st.error(f"❌ 키워드 로드 오류: {e}")
        return get_default_keywords()

@st.cache_data
def load_graduation_requirements():
    """졸업요건(기준학번별 학점) 로드"""
    try:
        df = pd.read_excel('data/graduation_requirements.xlsx')
        return df
    except FileNotFoundError:
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 졸업요건 로드 오류: {e}")
        return pd.DataFrame()

@st.cache_data
def load_primary_requirements():
    """본전공 이수요건 데이터 로드"""
    try:
        df = pd.read_excel('data/primary_requirements.xlsx')
        if not df.empty:
            cols = ['전공명', '구분']
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
        return df
    except:
        return pd.DataFrame()

@st.cache_data
def load_majors_info():
    """전공 정보 로드 (연락처, 홈페이지 포함)"""
    try:
        df = pd.read_excel('data/majors_info.xlsx')
        return df
    except FileNotFoundError:
        st.warning("⚠️ data/majors_info.xlsx 파일을 찾을 수 없습니다.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ 전공 정보 로드 오류: {e}")
        return pd.DataFrame()


def get_default_keywords():
    """기본 키워드 데이터"""
    return [
        {"키워드": "복수전공", "타입": "제도", "연결정보": "복수전공"},
        {"키워드": "복전", "타입": "제도", "연결정보": "복수전공"},
        {"키워드": "부전공", "타입": "제도", "연결정보": "부전공"},
        {"키워드": "부전", "타입": "제도", "연결정보": "부전공"},
        {"키워드": "연계전공", "타입": "제도", "연결정보": "연계전공"},
        {"키워드": "융합전공", "타입": "제도", "연결정보": "융합전공"},
        {"키워드": "융합부전공", "타입": "제도", "연결정보": "융합부전공"},
        {"키워드": "마이크로디그리", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "마디", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "MD", "타입": "제도", "연결정보": "마이크로디그리"},
        {"키워드": "학점", "타입": "주제", "연결정보": "학점정보"},
        {"키워드": "이수학점", "타입": "주제", "연결정보": "학점정보"},
        {"키워드": "신청", "타입": "주제", "연결정보": "신청정보"},
        {"키워드": "지원", "타입": "주제", "연결정보": "신청정보"},
        {"키워드": "비교", "타입": "주제", "연결정보": "비교표"},
        {"키워드": "차이", "타입": "주제", "연결정보": "비교표"},
        {"키워드": "졸업", "타입": "주제", "연결정보": "졸업요건"},
        {"키워드": "졸업인증", "타입": "주제", "연결정보": "졸업요건"},
        {"키워드": "졸업시험", "타입": "주제", "연결정보": "졸업요건"},
    ]

def get_sample_programs():
    """샘플 제도 데이터"""
    return {
        "복수전공": {
            "description": "주전공 외에 다른 전공을 추가로 이수하여 2개의 학위를 취득하는 제도",
            "credits_general": "-",
            "credits_major": "36학점 이상",
            "graduation_certification": "불필요",
            "graduation_exam": "불필요",
            "qualification": "2학년 이상, 평점 2.0 이상",
            "degree": "2개 학위 수여",
            "difficulty": "★★★★☆",
            "features": ["졸업 시 2개 학위 취득", "취업 시 경쟁력 강화", "학점 부담 높음"],
            "notes": ""
        },
        "부전공": {
            "description": "주전공 외에 다른 전공의 기초과목을 이수하는 제도",
            "credits_general": "-",
            "credits_major": "21학점 이상",
            "graduation_certification": "불필요",
            "graduation_exam": "불필요",
            "qualification": "2학년 이상",
            "degree": "주전공 학위 (부전공 표기)",
            "difficulty": "★★☆☆☆",
            "features": ["학점 부담 적음", "학위증에 부전공 표기"],
            "notes": ""
        }
    }

def get_sample_faq():
    """샘플 FAQ 데이터"""
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
KEYWORDS_DATA = load_keywords()
GRAD_REQUIREMENTS = load_graduation_requirements()
PRIMARY_REQUIREMENTS = load_primary_requirements()
MAJORS_INFO = load_majors_info()  # 🆕 전공 정보 로드


# === [핵심] AI 지식 검색 함수 (RAG) ===
def get_ai_context(user_input):
    context = ""
    user_input_clean = user_input.replace(" ", "").lower()

    # 1. 어떤 다전공 제도에 관심이 있는지 파악 (복수, 부, 융합 등)
    target_program = None
    for p in ["복수전공", "부전공", "융합전공", "융합부전공", "연계전공", "마이크로디그리"]:
        if p in user_input_clean or p[:2] in user_input_clean:
            target_program = p
            break

    # 2. 본전공 이수요건 변동 정보 검색 (PRIMARY_REQUIREMENTS 활용)
    if not PRIMARY_REQUIREMENTS.empty:
        # 전공 핵심어 추출 (예: 경영학전공 -> 경영)
        root_input = re.sub(r'(전공|학과|학부|의|신청|학점|어떻게|변해|알려줘|추천)', '', user_input_clean)
        
        # 전공명 매칭
        matched_primary = []
        for m in PRIMARY_REQUIREMENTS['전공명'].unique():
            if root_input in str(m).lower() or str(m).lower().replace("전공","") in root_input:
                matched_primary.append(m)
        
        if matched_primary:
            for m in matched_primary[:1]: # 가장 유사한 전공 하나 선택
                df_major = PRIMARY_REQUIREMENTS[PRIMARY_REQUIREMENTS['전공명'] == m]
                
                # [중요] 해당 전공의 모든 이수 요건(단일전공 포함)을 다 AI에게 줍니다.
                # 그래야 AI가 '단일전공'과 '복수전공'을 비교해서 설명할 수 있습니다.
                context += f"### [{m}] 본전공 이수학점 상세 기준\n"
                for _, row in df_major.iterrows():
                    context += f"- 구분: {row['구분']}\n"
                    context += f"  * 본전공 전필: {row.get('본전공_전필', 0)}학점\n"
                    context += f"  * 본전공 전선: {row.get('본전공_전선', 0)}학점\n"
                    context += f"  * 본전공 총합: {row.get('본전공_계', 0)}학점\n"
                context += "\n"

    # 1. 제도 카테고리 감지 (리스트를 뽑기 위한 키워드)
    # 사용자가 '융합전공 종류', '마디 리스트' 등을 물어볼 때 대응
    categories = {
        "융합전공": ["융합전공", "융합"],
        "부전공": ["부전공"],
        "복수전공": ["복수전공", "복전"],
        "마이크로디그리": ["마이크로디그리", "마디", "소단위", "md"],
        "연계전공": ["연계전공", "연계"]
    }

    target_category = None
    for cat_name, keywords in categories.items():
        if any(kw in user_input_clean for kw in keywords):
            target_category = cat_name
            # 사용자가 리스트를 원할 경우를 대비해 해당 카테고리 전체를 긁어옴
            if not MAJORS_INFO.empty and '제도유형' in MAJORS_INFO.columns:
                # '제도유형' 컬럼에 해당 카테고리명이 포함된 전공들 추출
                matched_rows = MAJORS_INFO[MAJORS_INFO['제도유형'].str.contains(cat_name, na=False)]
                if not matched_rows.empty:
                    major_list = matched_rows['전공명'].tolist()
                    context += f"[{cat_name} 전체 목록]\n- 현재 운영 중인 전공: {', '.join(major_list)}\n"
                    context += "(이 리스트를 학생에게 모두 나열하며 안내해주세요.)\n\n"

    # [의도 파악용 키워드]
    is_contact_query = any(w in user_input_clean for w in ["연락처", "사무실", "위치", "번호"])
    is_major_list_query = any(w in user_input_clean for w in ["전공", "종류", "리스트", "뭐있어"])
    is_apply_query = any(w in user_input_clean for w in ["신청", "기간", "절차", "방법", "언제"])
    
    # 1. 특정 전공 매칭 시도
    root_input = re.sub(r'(전공|학과|학부|의|과목|학년|리스트|추천|해줘|알려줘|뭐있어|설명|연락처|위치|사무실)', '', user_input_clean)

    if len(root_input) >= 2: # 최소 2글자 이상일 때만 상세 검색
        matched_majors = set()
        if not MAJORS_INFO.empty:
           for m in MAJORS_INFO['전공명'].unique():
                if root_input in str(m).lower() or str(m).lower().replace("전공","") in root_input:
                    matched_majors.add(str(m))
       
    for major in list(matched_majors)[:2]:
            m_info = MAJORS_INFO[MAJORS_INFO['전공명'] == major]
            if not m_info.empty:
                row = m_info.iloc[0]
                context += f"[{major} 상세정보]\n- 연락처: {row.get('연락처','-')}\n- 위치: {row.get('위치','-')}\n- 소개: {row.get('전공설명','-')}\n\n"

    # 2. 데이터 수집
    if matched_majors:
        # 특정 전공이 매칭된 경우 (상세 정보 제공)
        for major in list(matched_majors)[:2]:
            m_info = MAJORS_INFO[MAJORS_INFO['전공명'] == major]
            if not m_info.empty:
                row = m_info.iloc[0]
                context += f"[{major} 정보]\n- 연락처: {row.get('연락처','-')}\n- 위치: {row.get('위치','-')}\n- 소개: {row.get('전공설명','-')}\n\n"
    
    # [핵심 수정] 특정 전공이 없어도 범용 질문이면 '맛보기' 데이터 주입
    elif is_contact_query:
        context += "[주요 전공 연락처 맛보기]\n"
        # 상위 5개 전공 정보를 미리 줍니다.
        for _, row in MAJORS_INFO.head(5).iterrows():
            context += f"- {row['전공명']}: {row.get('연락처','-')} ({row.get('위치','-')})\n"
        context += f"\n[전체 전공 목록]: {', '.join(all_majors[:15])}... 등\n"

    # 2. 학년 파악 (1~4학년)
    target_year = None
    for i in range(1, 5):
        if f"{i}학년" in user_input_clean or str(i) in user_input_clean:
            target_year = i
            break
    
    # 3. 전공 매칭 로직 (중복 제거를 위해 set 사용)
    matched_majors = set()
    if not COURSES_DATA.empty:
        all_majors = COURSES_DATA['전공명'].unique()
        for m in all_majors:
            m_str = str(m)
            m_clean = m_str.replace(" ", "").lower()
            m_root = re.sub(r'(전공|학과|학부)', '', m_clean)
            
            # 검색어가 전공명에 포함되거나, 전공 핵심어가 검색어에 포함되는 경우 매칭
            if root_input in m_clean or m_root in root_input:
                matched_majors.add(m_str)

    # 4. 수집된 정보를 바탕으로 Context 구성
    if matched_majors:
        # 후보군 리스트 생성
        context += f"[검색된 전공 후보군: {', '.join(matched_majors)}]\n\n"
        
        # 각 전공별 상세 정보 및 과목 추출
        for major in list(matched_majors)[:2]: # 토큰 절약을 위해 최대 2개 전공만 상세 안내
            # A. 전공 기본 정보 (연락처, 설명 등)
            if not MAJORS_INFO.empty:
                m_info = MAJORS_INFO[MAJORS_INFO['전공명'] == major]
                if not m_info.empty:
                    row = m_info.iloc[0]
                    context += f"[{major} 상세 정보]\n- 소개: {row.get('전공설명','-')}\n- 연락처: {row.get('연락처','-')}\n- 위치: {row.get('위치','-')}\n"

            # B. 전공 과목 정보
            major_courses = COURSES_DATA[COURSES_DATA['전공명'] == major]
            if target_year:
                major_courses = major_courses[major_courses['학년'] == target_year]
                context += f"[{major} {target_year}학년 과목 리스트]\n"
            else:
                context += f"[{major} 주요 과목 리스트]\n"
            
            # 주요 과목 15개까지만 출력
            for _, row in major_courses.head(15).iterrows():
                context += f"- {row['학년']}학년 {row['학기']}학기: [{row['이수구분']}] {row['과목명']} ({row['학점']}학점)\n"
            context += "\n"
    else:
        # 매칭된 전공이 없을 때
        if len(root_input) > 1:
            context += f"[안내] 입력하신 '{root_input}'와 일치하는 전공을 찾지 못했습니다. 학생에게 정확한 전공명을 물어봐주세요.\n"

    # 6. FAQ 검색 (기존 중복 방지 로직 유지)
    if FAQ_DATA:
        added_faqs = set()
        # A. 사용자가 '신청'을 물어보면 '신청'이 포함된 모든 FAQ를 우선 수집
        if is_apply_query:
            for faq in FAQ_DATA:
                if "신청" in str(faq['질문']) or "기간" in str(faq['질문']):
                    context += f"[학사 안내: 신청 관련]\nQ: {faq['질문']}\nA: {faq['답변']}\n\n"
                    added_faqs.add(faq['질문'])

        # B. 일반 키워드 매칭
        for faq in FAQ_DATA:
            if faq['질문'] not in added_faqs:
                if user_input_clean in str(faq['질문']).lower() or user_input_clean in str(faq['답변']).lower():
                    context += f"[참고 FAQ]\nQ: {faq['질문']}\nA: {faq['답변']}\n\n"
                    added_faqs.add(faq['질문'])

    # 3. 제도 정보 검색 (PROGRAM_INFO)
    for p_name, p_info in PROGRAM_INFO.items():
        if p_name in user_input_clean:
            context += f"### [{p_name}] 제도 자체 이수 기준\n"
            context += f"- 설명: {p_info['description']}\n"
            context += f"- 이 제도 이수를 위해 필요한 학점: {p_info['credits_multi']}\n\n"

    return context

# === [핵심] Gemini API 답변 생성 ===
def generate_ai_response(user_input, chat_history):
    """Gemini API를 사용하여 답변 생성"""
    
    # 1. 엑셀에서 관련 지식 추출
    context = get_ai_context(user_input)
    
    # 2. 대화 기록 요약 (최근 3개만)
    history_text = ""
    for chat in chat_history[-3:]:
        history_text += f"{chat['role']}: {chat['content']}\n"

    # 3. 시스템 프롬프트 (AI의 성격과 규칙 설정)
    prompt = f"""
    당신은 '한경국립대학교'의 유연학사제도(다전공) 안내 전문 AI 상담원입니다.
    질문에 대해 아래 제공된 [학사 데이터]만을 근거로 답변하세요.
    학생이 다전공 신청에 대해 물으면, 다전공 학점뿐만 아니라 [본전공 학점 변동] 정보도 반드시 확인해서 알려주세요.
    
    [학사 데이터]
    {context if context else "검색 결과 없음"}

    [대화 기록]
    {history_text}

    [규칙]
    1. 반드시 제공된 [학사 데이터]를 최우선으로 참고하여 답변하세요.
    2. 학생이 특정 전공의 과목을 물어보거나 추천을 요청하면, 데이터에 있는 과목명을 언급하며 추천 이유를 짧게 설명하세요.
    3. '자료가 부족하여 제공해 드리기 어렵습니다', '학사 시스템 내 별도의 페이지에서 확인하라', '홈페이지를 참고하라', '포털에서 조회하라'는 식의 무책임하거나 모호한 안내는 절대 하지 마세요.
    4. 데이터에 없는 내용을 답변할 때는 '제가 가진 자료에는 없지만 일반적인 내용은 이렇습니다'라고 밝히고, 정확한 확인은 해당 전공 또는 학사지원팀에 문의하라고 안내하세요.
    5. 과목 리스트, 수강해야할 과목 등 확인은 왼쪽 메뉴의 '다전공 제도 안내'에서 확인하라고 안내하세요.
    6. 말투는 친절하고 명확하게 '습니다'체를 사용하세요.
    7. 중요한 수치(학점 등)는 강조(**) 표시를 하세요.
    8. 답변 끝에는 연관된 키워드(예: #복수전공 #신청기간)를 2~3개 달아주세요.
    9. 전공명이 모호한 경우(예: '행정'만 입력): 
       - "혹시 '행정학전공'을 찾으시는 걸까요?"와 같이 후보군 중에서 가장 가능성 높은 전공을 되물어보세요.
       - 데이터에 검색된 후보군({context.split(']')[0] if ']' in context else ''})이 있다면 이를 리스트로 보여주세요.
    10. 질문 가이드 제공:
       - 답변 마지막에 항상 "💡 더 정확한 정보를 원하시면 '행정학전공 2학년 과목 알려줘'와 같이 [전공명 + 학년]을 포함하여 질문해 주세요!"라는 가이드를 넣으세요.
    11. 과목 추천:
       - 데이터에 과목 정보가 있다면 되묻는 동시에 "우선 찾으시는 전공일 것으로 예상되는 {context.split('[')[1].split(' ')[0] if '[' in context else '해당 전공'}의 과목을 안내해 드립니다"라며 맛보기 정보를 제공하세요.
    12. 친절도: 학생을 대하듯 친절하고 따뜻하게 답변하세요.
    13. 학생이 질문한 내용에 대해 데이터가 부족하다면, 질문 예시(예: 전공명과 학년을 함께 말씀해 주세요)를 참고하여 다시 질문하도록 친절하게 유도해줘.
    14. 질문 예시(버튼)를 누른 경우처럼 질문이 조금 포괄적이더라도, "구체적으로 말해달라"는 답변부터 하지 마세요.
    15. 데이터에 있는 정보(연락처 맛보기, 전공 리스트 등)를 활용하여 일단 아는 범위 내에서 최대한 풍부하게 답변하세요.
    16. 연락처를 물으면 표(Table) 형식을 사용하여 깔끔하게 정리해 보여주세요.
    17. 정보가 많아 리스트를 보여준 후에는, "더 궁금한 특정 전공이 있다면 이름을 말씀해 주세요!"라고 자연스럽게 유도하세요.
    18. 만약 특정 전공의 신청 절차가 데이터에 없다면, 제공된 [데이터] 중 '다전공 신청'이나 '일반적인 신청 기간' 정보를 활용하여 "공통적으로 다전공 신청은 매년 4월, 10월경에 진행됩니다"와 같이 아는 범위 내에서 최대한 답변하세요.
    19. 데이터에 신청 기간 정보가 조금이라도 있다면 그것을 최우선으로 안내하세요.
    20. 정보가 정 부족하다면 답변 끝에 "더 상세한 개인별 상황은 학사지원팀(031-670-5035)에 문의하면 정확히 확인할 수 있습니다"라고 덧붙이세요.
    21. 데이터에 [본전공 학점 변동 정보]가 포함되어 있다면, 이를 강조해서 안내하세요. 
    22. 예: "행정학전공 학생이 복수전공을 신청하면, 본전공 이수 학점이 기존 70학점에서 45학점으로 줄어들어 부담이 적어집니다!"와 같은 방식으로 설명하세요.
    23. 만약 사용자의 전공이 무엇인지 모른다면, "본전공에 따라 다전공 신청 시 본전공 이수 학점이 줄어들 수 있으니, 본전공 이름을 말씀해주시면 더 정확히 안내해 드릴게요."라고 친절히 되물으세요.
    24. 학생이 특정 전공(예: 경영학전공)에서 다전공(예: 복수전공)을 할 때의 학점 변화를 물으면:
       - 데이터에 있는 '구분: 단일전공'일 때의 학점과 '구분: 복수전공'일 때의 학점을 찾아 서로 비교해주세요.
       - "단일전공 시에는 본전공을 00학점 들어야 하지만, 복수전공을 신청하면 00학점으로 줄어듭니다"라고 명확히 말하세요.
    25. 절대로 "구체적인 정보가 포함되어 있지 않습니다"라는 말을 먼저 하지 마세요. 데이터에 '구분'별 학점이 있다면 그것이 바로 그 정보입니다.
    26. 정보를 표(Table) 형태로 정리해서 보여주면 학생이 이해하기 쉽습니다.
    27. 데이터에 본전공 이름은 있는데 신청하려는 제도(예: 융합전공)에 대한 행이 없다면, "단일전공 기준은 이렇습니다. 다전공 신청 시 변동 수치는 학과 사무실에 확인이 필요합니다."라고 안내하세요.

    질문: {user_input}
    """

    try:
        # 2.0이나 2.5가 아닌 가장 대중적인 1.5 Flash를 사용합니다.
        response = client.models.generate_content(
            model='gemini-flash-latest', # <--- 이 이름으로 변경
            contents=prompt
        )
        if response and response.text:
            return response.text, "ai_generated"     
    except Exception as e:
        return str(e), "error"

# === 메인 화면 로직 수정 ===
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "안녕하세요! 한경국립대학교 다전공 안내 AI 비서입니다. 궁금한 점을 물어보세요! 🎓", "response_type": "greeting"}
    ]

# === 키워드 검색 함수 ===
def search_by_keyword(user_input):
    """키워드 기반 검색 (최우선)"""
    user_input_lower = user_input.lower()
    
    matched_keywords = []
    
    for keyword_data in KEYWORDS_DATA:
        keyword = keyword_data['키워드'].lower()
        
        if keyword in user_input_lower:
            matched_keywords.append(keyword_data)
    
    if matched_keywords:
        matched_keywords.sort(key=lambda x: len(x['키워드']), reverse=True)
        return matched_keywords[0]
    
    return None

def find_majors_with_details(user_input):
    """
    단어만 입력해도 전공명/키워드와 매칭하여 상세 정보를 반환
    """
    if MAJORS_INFO.empty:
        return []
    
    # 1. 입력값 정제 (공백 제거)
    user_input_clean = user_input.replace(" ", "").lower()
    
    # 입력값이 너무 짧으면(1글자) 검색 품질을 위해 제외 (예: '학', '과' 등)
    if len(user_input_clean) < 2:
        return []

    results = []
    
    for _, row in MAJORS_INFO.iterrows():
        # 데이터 정제
        major_name = str(row['전공명']).strip()
        major_clean = major_name.replace(" ", "").lower()
        
        # '전공', '학과', '학부'를 뗀 핵심 단어 추출 (예: 경영학전공 -> 경영학)
        core_name = major_clean.replace("전공", "").replace("학과", "").replace("학부", "")
        
        # 키워드 가져오기
        keywords = str(row.get('관심분야키워드', '')).lower()
        keyword_list = [k.strip().replace(" ", "") for k in keywords.split(',')]
        
        # === 매칭 로직 ===
        match_found = False
        priority = 0
        
        # Case A: 전공명에 입력어가 포함됨 (예: 입력 '경영' -> 데이터 '경영전공')
        if user_input_clean in major_clean: 
            match_found = True
            priority = 3  # 가장 높은 우선순위
            
        # Case B: 핵심 단어가 입력어와 같음 (예: 입력 '경영' -> 데이터 '경영학'의 핵심 '경영')
        elif core_name in user_input_clean:
            match_found = True
            priority = 2
            
        # Case C: 키워드 매칭 (예: 입력 '회계' -> 키워드 '회계')
        elif any(user_input_clean in k for k in keyword_list if k):
            match_found = True
            priority = 1

        if match_found:
            results.append({
                'major': major_name,
                'description': row.get('전공설명', '설명 없음'),
                'contact': row.get('연락처', '-'),
                'homepage': row.get('홈페이지', '-'),
                'location': row.get('위치', '-'),
                'program_types': row.get('제도유형', '-'),
                'priority': priority
            })
    
    # 우선순위 높음 -> 이름 짧은 순(정확도 높을 확률)으로 정렬
    results.sort(key=lambda x: (-x['priority'], len(x['major'])))
    
    return results


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
    """유사한 FAQ 찾기"""
    vectorizer, faq_vectors, questions = create_faq_vectorizer()
    
    if vectorizer is None or not questions:
        return None
    
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    
    max_similarity_idx = np.argmax(similarities)
    max_similarity = similarities[max_similarity_idx]
    
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
    
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            results.append({
                'faq': FAQ_DATA[idx],
                'similarity': similarities[idx]
            })
    
    return results

def find_similar_program(user_input):
    """제도명 유사도 검색"""
    program_names = list(PROGRAM_INFO.keys())
    
    for program in program_names:
        if program in user_input:
            return program
    
    for program in program_names:
        if any(word in user_input for word in program.split()):
            return program
    
    return None

# === 🆕 관심분야 기반 전공 추천 함수 ===
def recommend_majors_by_interest(user_input):
    """관심분야 키워드 매칭 로직 개선"""
    # 1. 데이터 로드 확인
    if MAJORS_INFO.empty:
        return []
    
    # 2. 필수 컬럼 확인 (컬럼명이 다를 경우를 대비해 유연하게 처리 가능)
    if '관심분야키워드' not in MAJORS_INFO.columns:
        # 컬럼명이 다를 경우 수동으로 매핑하거나 빈 리스트 반환
        return []

    user_input_lower = user_input.lower()
    recommendations = []
    
    for _, row in MAJORS_INFO.iterrows():
        # 데이터 전처리 (NaN 처리 및 문자열 변환)
        raw_keywords = str(row.get('관심분야키워드', ''))
        if raw_keywords == 'nan' or not raw_keywords.strip():
            continue
            
        # 콤마(,) 기준으로 나누고 공백 제거
        keywords_list = [k.strip().lower() for k in raw_keywords.split(',')]
        
        # 3. 매칭 검사: 입력 문장에 키워드가 포함되어 있는지 확인
        # (예: 입력 "인공지능 배우고 싶어" -> 키워드 "인공지능" 매칭)
        matched = [k for k in keywords_list if k in user_input_lower]
        
        if matched:
            recommendations.append({
                'major': row['전공명'],
                'description': row.get('전공설명', '설명 없음'),
                'program_types': row.get('제도유형', '-'),
                'match_score': len(matched), # 매칭된 키워드 개수로 점수 산정
                'matched_keywords': matched,
                'contact': row.get('연락처', '-'),
                'homepage': row.get('홈페이지', '-')
            })
    
    # 매칭 점수가 높은 순으로 정렬 후 상위 5개 반환
    recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    return recommendations[:5]

def display_major_info(major_name):
    """특정 전공의 연락처/홈페이지 정보 표시"""
    if MAJORS_INFO.empty:
        return "전공 정보를 불러올 수 없습니다."
    
    major_data = MAJORS_INFO[MAJORS_INFO['전공명'] == major_name]
    
    if major_data.empty:
        return f"'{major_name}' 전공 정보를 찾을 수 없습니다."
    
    row = major_data.iloc[0]
    
    response = f"**{major_name} 📞**\n\n"
    response += f"**📝 소개:** {row['전공설명']}\n\n"
    response += f"**📚 이수 가능 다전공 제도:** {row['제도유형']}\n\n"
    response += f"**📞 연락처:** {row['연락처']}\n\n"
    
    if pd.notna(row.get('홈페이지')) and row['홈페이지'] != '-':
        response += f"**🌐 홈페이지:** {row['홈페이지']}\n\n"
    
    if pd.notna(row.get('위치')) and row['위치'] != '-':
        response += f"**📍 위치:** {row['위치']}\n\n"
    
    return response


# === 이미지 표시 함수 ===
def display_curriculum_image(major, program_type):
    """이수체계도 또는 안내 이미지 표시"""
    result = CURRICULUM_MAPPING[
        (CURRICULUM_MAPPING['전공명'] == major) & 
        (CURRICULUM_MAPPING['제도유형'] == program_type)
    ]
    
    if not result.empty:
        raw_filenames = str(result.iloc[0]['파일명'])
        filenames = [f.strip() for f in raw_filenames.split(',')]
        
        if len(filenames) > 1:
            cols = st.columns(len(filenames)) 
            for idx, filename in enumerate(filenames):
                image_path = f"images/curriculum/{filename}"
                with cols[idx]:
                    if os.path.exists(image_path):
                        st.image(image_path, caption=f"{major} 안내-{idx+1}", use_container_width=True)
                    else:
                        st.warning(f"⚠️ 이미지 파일 없음: {filename}")
            return True
            
        else:
            filename = filenames[0]
            image_path = f"images/curriculum/{filename}"
            
            if os.path.exists(image_path):
                is_micro = "소단위전공과정(마이크로디그리)" in program_type or "마이크로디그" in program_type
                caption_text = f"{major} 안내 이미지" if is_micro else f"{major} 이수체계도"
                
                if is_micro:
                    col1, col2, col3 = st.columns([1, 2, 1]) 
                    with col2:
                        st.image(image_path, caption=caption_text, use_container_width=True)
                else:
                    st.image(image_path, caption=caption_text, use_container_width=True)
                
                return True
            else:
                st.warning(f"⚠️ 이미지를 찾을 수 없습니다: {image_path}")
                return False
    else:
        if "소단위전공과정(마이크로디그리)" not in program_type:
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
        st.subheader(f"📚 {major} 편성 교과목(2025학년도 교육과정)")       
        
        if "소단위전공과정(마이크로디그리)" in program_type:
            semesters = sorted(courses['학기'].unique())
            
            for semester in semesters:
                st.markdown(f"#### {int(semester)}학기")
                
                semester_courses = courses[courses['학기'] == semester]
                
                for _, course in semester_courses.iterrows():
                    division = course['이수구분']
                    course_name = course['과목명']
                    credits = int(course['학점'])
                    
                    if division in ['전필', '필수']:
                        badge_color = "🔴"
                    elif division in ['전선', '선택']:
                        badge_color = "🟢"
                    else:
                        badge_color = "🔵"
                    
                    st.write(f"{badge_color} **[{division}]** {course_name} ({credits}학점)")
                
                st.write("")
                
        else:
            years = sorted([int(y) for y in courses['학년'].unique() if pd.notna(y)])
            
            if len(years) > 0:
                tabs = st.tabs([f"{year}학년" for year in years])
                
                for idx, year in enumerate(years):
                    with tabs[idx]:
                        year_courses = courses[courses['학년'] == year]
                        semesters = sorted(year_courses['학기'].unique())
                        
                        for semester in semesters:
                            st.write(f"**{int(semester)}학기**")
                            semester_courses = year_courses[year_courses['학기'] == semester]
                            
                            for _, course in semester_courses.iterrows():
                                division = course['이수구분']
                                course_name = course['과목명']
                                credits = int(course['학점'])
                                
                                if division in ['전필', '필수']:
                                    badge_color = "🔴"
                                elif division in ['전선', '선택']:
                                    badge_color = "🟢"
                                else:
                                    badge_color = "🔵"
                                
                                st.write(f"{badge_color} **[{division}]** {course_name} ({credits}학점)")
                            
                            st.write("")
               
        return True
    else:
        return False

# === 비교표 생성 ===
def create_comparison_table():
    data = {
        "제도": list(PROGRAM_INFO.keys()),
        "이수학점(교양)": [info["credits_general"] for info in PROGRAM_INFO.values()],
        "원전공 이수학점": [info["credits_primary"] for info in PROGRAM_INFO.values()],
        "다전공 이수학점": [info["credits_multi"] for info in PROGRAM_INFO.values()],
        "졸업인증": [info["graduation_certification"] for info in PROGRAM_INFO.values()],
        "졸업시험": [info["graduation_exam"] for info in PROGRAM_INFO.values()],
        "학위표기": [info["degree"] for info in PROGRAM_INFO.values()],
        "난이도": [info["difficulty"] for info in PROGRAM_INFO.values()],
        "신청자격": [info["qualification"] for info in PROGRAM_INFO.values()]
    }
    return pd.DataFrame(data)

# === 졸업학점 계산 및 다전공 추천 함수 ===
def calculate_remaining_credits(primary_major, admission_year, completed_required, completed_elective):
    """본전공 졸업요건 대비 남은 학점 계산"""
    if PRIMARY_REQUIREMENTS.empty:
        return None
    
    pri_data = PRIMARY_REQUIREMENTS[PRIMARY_REQUIREMENTS['전공명'] == primary_major].copy()
    pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
    pri_valid = pri_data[pri_data['기준학번'] <= admission_year]
    
    if pri_valid.empty:
        return None
    
    # 단일전공 기준 찾기
    pri_valid = pri_valid.sort_values('기준학번', ascending=False)
    single_major_row = None
    
    for _, row in pri_valid.iterrows():
        if '단일전공' in str(row['구분']) or pd.isna(row['구분']):
            single_major_row = row
            break
    
    if single_major_row is None:
        single_major_row = pri_valid.iloc[0]
    
    required_credits = int(single_major_row['본전공_전필'])
    elective_credits = int(single_major_row['본전공_전선'])
    total_required = required_credits + elective_credits
    
    remaining_required = max(0, required_credits - completed_required)
    remaining_elective = max(0, elective_credits - completed_elective)
    total_remaining = remaining_required + remaining_elective
    
    completed_total = completed_required + completed_elective
    progress = (completed_total / total_required * 100) if total_required > 0 else 0
    
    return {
        'required_credits': required_credits,
        'elective_credits': elective_credits,
        'total_required': total_required,
        'remaining_required': remaining_required,
        'remaining_elective': remaining_elective,
        'total_remaining': total_remaining,
        'completed_total': completed_total,
        'progress': progress
    }

def recommend_programs(primary_major, admission_year, current_grade, completed_required, completed_elective):
    """다전공 추천 시스템"""
    recommendations = []
    
    # 현재 학년에서 남은 학기 계산 (8학기 기준)
    remaining_semesters = (8 - (current_grade * 2 - 2)) if current_grade <= 4 else 2
    
    # 본전공 남은 학점
    primary_result = calculate_remaining_credits(primary_major, admission_year, completed_required, completed_elective)
    
    if primary_result is None:
        return []
    
    primary_remaining = primary_result['total_remaining']
    
    # 각 제도별 분석
    for program_name, program_info in PROGRAM_INFO.items():
        # 학점 요구사항 파싱
        major_credits_str = program_info['credits_multi']
        
        # 숫자 추출
        credits_match = re.search(r'(\d+)', major_credits_str)
        if not credits_match:
            continue
        
        required_credits = int(credits_match.group(1))
        
        # 난이도 점수
        difficulty = program_info['difficulty'].count('★')
        
        # 본전공 변동 학점 확인
        additional_primary_credits = 0
        if not PRIMARY_REQUIREMENTS.empty and primary_major:
            pri_data = PRIMARY_REQUIREMENTS[PRIMARY_REQUIREMENTS['전공명'] == primary_major].copy()
            pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
            pri_valid = pri_data[pri_data['기준학번'] <= admission_year]
            
            if not pri_valid.empty:
                pri_valid = pri_valid.sort_values('기준학번', ascending=False)
                
                for _, p_row in pri_valid.iterrows():
                    if program_name in str(p_row['구분']):
                        single_total = primary_result['total_required']
                        modified_total = int(p_row['본전공_계'])
                        additional_primary_credits = max(0, modified_total - single_total)
                        break
        
        # 총 필요 학점
        total_needed = required_credits + additional_primary_credits
        
        # 학기당 평균 이수 가능 학점
        available_credits_per_semester = 18
        total_available_credits = remaining_semesters * available_credits_per_semester
        
        # 본전공에 쓸 학점 제외
        net_available = total_available_credits - primary_remaining
        
        # 가능성 점수 계산
        if net_available <= 0:
            feasibility = "매우 낮음"
            score = 0
        elif total_needed <= net_available * 0.6:
            feasibility = "높음"
            score = 90 - (difficulty * 5)
        elif total_needed <= net_available * 0.85:
            feasibility = "보통"
            score = 70 - (difficulty * 5)
        elif total_needed <= net_available:
            feasibility = "낮음"
            score = 50 - (difficulty * 5)
        else:
            feasibility = "매우 낮음"
            score = max(0, 30 - (difficulty * 5))
        
        # 이유 생성
        reasons = []
        if feasibility in ["높음", "보통"]:
            reasons.append(f"✅ 남은 학기 내 이수 가능 ({remaining_semesters}학기)")
            if difficulty <= 2:
                reasons.append("✅ 낮은 난이도")
            if additional_primary_credits == 0:
                reasons.append("✅ 본전공 학점 변동 없음")
        else:
            if total_needed > net_available:
                reasons.append(f"⚠️ 필요 학점({total_needed})이 여유 학점({int(net_available)})보다 많음")
            if difficulty >= 4:
                reasons.append("⚠️ 높은 난이도")
            if additional_primary_credits > 0:
                reasons.append(f"⚠️ 본전공 학점 {additional_primary_credits}학점 추가 필요")
        
        recommendations.append({
            'program': program_name,
            'feasibility': feasibility,
            'score': score,
            'required_credits': required_credits,
            'additional_primary_credits': additional_primary_credits,
            'total_needed': total_needed,
            'net_available': int(net_available),
            'difficulty': difficulty,
            'reasons': reasons,
            'description': program_info['description'],
            'degree': program_info['degree']
        })
    
    # 점수순 정렬
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations

def generate_action_plan(recommendation, current_grade, remaining_semesters):
    """구체적인 액션 플랜 생성"""
    program = recommendation['program']
    feasibility = recommendation['feasibility']
    
    plan = []
    
    if feasibility == "높음":
        plan.append(f"**1단계: 지금 바로 신청 준비 🚀**")
        plan.append(f"- {program} 신청 자격 확인 (평점 등)")
        plan.append(f"- 다음 신청 기간 체크 (학기 초/말)")
        plan.append(f"")
        plan.append(f"**2단계: 이수 계획 수립 📝**")
        plan.append(f"- 학기당 {recommendation['required_credits'] // remaining_semesters + 1}학점씩 이수")
        plan.append(f"- 전공필수 과목 우선 수강")
        plan.append(f"")
        plan.append(f"**3단계: 사전 준비 💪**")
        plan.append(f"- 해당 전공 교수님 상담 권장")
        plan.append(f"- 선배들의 이수 경험 참고")
    
    elif feasibility == "보통":
        plan.append(f"**1단계: 신중한 검토 필요 🤔**")
        plan.append(f"- 본전공 학점 이수 계획 먼저 확정")
        plan.append(f"- 학기당 수강 가능 학점 현실적으로 계산")
        plan.append(f"")
        plan.append(f"**2단계: 대안 고려 ⚖️**")
        plan.append(f"- 더 낮은 학점의 제도(부전공, 마이크로디그리) 검토")
        plan.append(f"- 계절학기 활용 가능성 확인")
        plan.append(f"")
        plan.append(f"**3단계: 상담 필수 📞**")
        plan.append(f"- 학사지원팀 상담으로 정확한 이수 가능성 확인")
        plan.append(f"- 지도교수님과 졸업 계획 논의")
    
    else:
        plan.append(f"**1단계: 현실적인 대안 검토 🔄**")
        plan.append(f"- 부전공(21학점) 또는 마이크로디그리(12~18학점) 추천")
        plan.append(f"- 졸업 후 추가 학기 고려 여부 판단")
        plan.append(f"")
        plan.append(f"**2단계: 학점 확보 전략 📚**")
        plan.append(f"- 계절학기 필수 활용")
        plan.append(f"- 학점 교류/교환학생 프로그램 검토")
        plan.append(f"")
        plan.append(f"**3단계: 전문가 상담 필수 ⚠️**")
        plan.append(f"- 학사지원팀에서 정확한 이수 가능성 확인")
        plan.append(f"- 다른 역량 개발 방안도 함께 논의")
    
    return "\n".join(plan)

# === 챗봇 응답 생성 ===
def generate_response(user_input):
    user_input_lower = user_input.lower()
    
    # 1. 인사
    if any(x in user_input_lower for x in ["안녕", "하이", "hello", "반가"]):
        return "안녕하세요! 👋 유연학사제도 안내 챗봇입니다. 궁금한 전공이나 제도를 물어보세요!", "greeting"

    # ====================================================
    # 2. [통합 검색] 전공/관심분야 검색 (최우선 처리)
    # "경영", "컴퓨터 연락처", "AI 추천" 등 모든 케이스를 여기서 처리
    # ====================================================
    search_results = find_majors_with_details(user_input)
    
    if search_results:
        response = f"**🔍 '{user_input}' 관련 전공 정보입니다.**\n\n"
        
        # 상위 3개만 표시
        for idx, info in enumerate(search_results[:3], 1):
            response += f"### {idx}. {info['major']}\n"
            
            # 소개 (설명이 없으면 생략)
            if info['description'] and info['description'] != '설명 없음':
                response += f"**📝 소개:** {info['description']}\n\n"
            
            # 연락처 (필수 정보)
            response += f"**📞 연락처:** {info['contact']}\n"
            
            # 홈페이지 (정보가 있는 경우만 표시)
            if info['homepage'] not in ['-', 'nan', None, '']:
                 response += f"**🌐 홈페이지:** [{info['homepage']}]({info['homepage']})\n"
            
            # 위치 (정보가 있는 경우만 표시)
            if info['location'] not in ['-', 'nan', None, '']:
                response += f"**📍 전공 사무실 위치:** {info['location']}\n"
            
            # 제도 유형
            response += f"\n**🎓 이수 가능 다전공:** {info['program_types']}\n"
            response += "\n"
            
        return response, "major_info"

    # ====================================================
    # 3. [예외 처리] 전공명 없이 '연락처'만 물어본 경우
    # 검색 결과가 없을 때만 실행됨 -> 전체 목록 제공
    # ====================================================
    if any(word in user_input_lower for word in ["연락처", "전화번호", "과사", "사무실"]):
        response = "**📞 전공별 연락처 안내**\n\n"
        response += "찾으시는 **전공명을 정확히 말씀해주시면** 해당 사무실 정보를 안내해드립니다.\n"
        response += "아래 목록에 있는 전공명을 입력해 보세요.\n\n"
        
        if not MAJORS_INFO.empty:
            # 1. 데이터 정리
            df_clean = MAJORS_INFO.dropna(subset=['전공명']).copy()
            df_clean['전공명'] = df_clean['전공명'].astype(str)
            
            # 2. 그룹 분리 로직 (마이크로디그리 vs 일반)
            try:
                is_md = df_clean['제도유형'].str.contains('마이크로|소단위', na=False) | \
                        df_clean['전공명'].str.contains('마이크로|소단위', na=False)
            except KeyError:
                is_md = df_clean['전공명'].str.contains('마이크로|소단위', na=False)

            general_majors = sorted(df_clean[~is_md]['전공명'].unique())
            md_majors = sorted(df_clean[is_md]['전공명'].unique())
            
            # 3. 일반 전공 출력
            response += "### 🏫 학부/전공\n"
            if general_majors:
                for i in range(0, len(general_majors), 3):
                    batch = general_majors[i:i+3]
                    response += " | ".join(batch) + "\n"
            
            # 4. 마이크로디그리 출력
            if md_majors:
                response += "\n### 🎓 소단위전공(마이크로디그리)\n"
                for i in range(0, len(md_majors), 2):
                    batch = md_majors[i:i+2]
                    response += " | ".join(batch) + "\n"
        
        return response, "contact_list"

    # ====================================================
    # 4. 제도 키워드 검색
    # ====================================================
    keyword_match = search_by_keyword(user_input)
    if keyword_match:
        keyword_type = keyword_match['타입']
        linked_info = keyword_match['연결정보']
        
        if keyword_type == "제도" and linked_info in PROGRAM_INFO:
            info = PROGRAM_INFO[linked_info]
            response = f"**{linked_info}** 📚\n\n"
            response += f"**설명:** {info['description']}\n\n"
            response += f"**📖 이수학점**\n"
            response += f"- 교양: {info['credits_general']}\n"
            response += f"- 원전공: {info['credits_primary']}\n\n"
            response += f"- 다전공: {info['credits_multi']}\n\n"
            response += f"**🎓 졸업 요건**\n"
            response += f"- 졸업인증: {info['graduation_certification']}\n"
            response += f"- 졸업시험: {info['graduation_exam']}\n\n"
            response += f"**✅ 신청자격:** {info['qualification']}\n"
            response += f"**📜 학위표기:** {info['degree']}\n"
            response += f"**♧ 난이도:** {info['difficulty']}\n\n"
            
            if info['features']:
                response += f"**✨ 특징:**\n"
                for feature in info['features']:
                    response += f"- {feature.strip()}\n"
            if info['notes']:
                response += f"\n**💡 기타:** {info['notes']}"
                
            response += f"\n\n_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
            return response, "program" # [수정] 올바른 response 리턴
        
        elif keyword_type == "주제":
            if linked_info == "학점정보":
                response = "**제도별 이수 학점** 📖\n\n"
                for program, info in PROGRAM_INFO.items():
                    response += f"**{program}**\n"
                    response += f"  - 교양: {info['credits_general']}\n"
                    response += f"  - 원전공: {info['credits_primary']}\n\n"
                    response += f"  - 다전공: {info['credits_multi']}\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "credits"
            
            elif linked_info == "신청정보":
                response = "**신청 관련 정보** 📝\n\n"
                response += "다전공 제도는 매 학기 초(4월, 10월), 학기말(6월, 12월)에 신청 가능합니다.\n\n"
                response += "자세한 내용은 '📚 다전공 제도 안내' 또는 '❓ FAQ' 메뉴'를 확인하시거나, - [📥 홈페이지 학사공지](https://www.hknu.ac.kr/kor/562/subview.do)\n를 참고해 주세요!\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "application"
            
            elif linked_info == "비교표":
                response = "각 제도의 비교는 왼쪽 사이드바의 '📚 다전공 제도 안내'에서 확인하실 수 있습니다!\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "comparison"
            
            elif linked_info == "졸업요건":
                response = "**제도별 졸업 요건** 🎓\n\n"
                for program, info in PROGRAM_INFO.items():
                    response += f"**{program}**\n"
                    response += f"  - 졸업인증: {info['graduation_certification']}\n"
                    response += f"  - 졸업시험: {info['graduation_exam']}\n\n"
                response += f"_🔍 키워드 '{keyword_match['키워드']}'로 검색됨_"
                return response, "graduation"
    
    # ====================================================
    # 5. FAQ 및 기타 로직
    # ====================================================
    
    # FAQ 유사도 검색
    similar_faq = find_similar_faq(user_input)
    if similar_faq:
        faq, similarity = similar_faq
        response = f"**Q. {faq['질문']}**\n\nA. {faq['답변']}\n\n"
        response += f"_💡 답변 신뢰도: {similarity*100:.0f}%_"
        return response, "faq"
    
    # 제도 설명 검색 (유사도)
    program = find_similar_program(user_input)
    if program:
        info = PROGRAM_INFO[program]
        response = f"**{program}** 📚\n\n"
        response += f"**설명:** {info['description']}\n..." # (길어서 생략, 필요한 경우 위와 동일하게 작성)
        return response, "program"
    
    # 비교 질문
    if any(word in user_input_lower for word in ["비교", "차이", "다른점", "vs"]):
        return "각 제도의 비교는 왼쪽 사이드바의 '📚 다전공 제도 안내'에서 확인하실 수 있습니다!", "comparison"
    
    # 학점 관련 (키워드 매칭 실패 시 백업)
    if any(word in user_input_lower for word in ["학점", "몇학점"]):
        response = "**제도별 이수 학점** 📖\n\n"
        for program, info in PROGRAM_INFO.items():
            response += f"**{program}**\n - 교양: {info['credits_general']}\n - 원전공: {info['credits_primary']}\n - 다전공: {info['credits_multi']}\n\n"
        return response, "credits"
    
    # 신청 관련 (백업)
    if any(word in user_input_lower for word in ["신청", "지원", "언제", "기간"]):
        return "매 학기 초(4월, 10월) 및 학기말(6월, 12월)에 신청 가능합니다.", "application"
    
    # 유사 질문 제안
    similar_faqs = get_top_similar_faqs(user_input, top_n=3)
    if similar_faqs:
        response = "정확히 일치하는 답변을 찾지 못했습니다. 😅\n\n**혹시 다음 질문 중 하나를 찾으셨나요?**\n\n"
        for i, item in enumerate(similar_faqs, 1):
            response += f"{i}. {item['faq']['질문']} _({item['similarity']*100:.0f}%)_\n"
        return response, "suggestion"
    
    # 완전 매칭 실패
    return "죄송합니다. 질문을 이해하지 못했습니다. 😅\n'경영'이나 '복수전공'처럼 핵심 단어로 질문해 보시겠어요?", "no_match"

# === 사이드바 ===
with st.sidebar:
    st.title("🎓 한경국립대 유연학사제도(다전공) 안내")
    
    # 관리자 모드 토글
    with st.expander("🔐 관리자 모드"):
        if not st.session_state.is_admin:
            admin_password = st.text_input("비밀번호", type="password", key="admin_login")
            if st.button("로그인"):
                if admin_password == ADMIN_PASSWORD:
                    st.session_state.is_admin = True
                    st.success("✅ 관리자 로그인 성공!")
                    st.rerun()
                else:
                    st.error("❌ 비밀번호가 틀렸습니다.")
        else:
            st.success("✅ 관리자 모드 활성화")
            if st.button("로그아웃"):
                st.session_state.is_admin = False
                st.rerun()
    
    st.divider()
    
    # 메뉴 선택
    if st.session_state.is_admin:
        menu = st.radio(
            "메뉴 선택",
            ["💬 챗봇", "📚 다전공 제도 안내", "❓ FAQ", "🔑 키워드 관리", "📊 피드백 통계"]
        )
    else:
        menu = st.radio(
            "메뉴 선택",
            ["💬 챗봇", "📚 다전공 제도 안내", "❓ FAQ"]
        )
    
    st.divider()
    
    st.subheader("빠른 질문")
    quick_questions = [
        "복수전공이 뭐야?",
        "부전공 학점은?",
        "신청은 언제 해?",
        "제도 비교해줘"
    ]
    
    for i, q in enumerate(quick_questions):
        if st.button(q, key=f"quick_q_{i}"):
            st.session_state.chat_history.append(
                {"role": "user", "content": q}
            )
            response, response_type = generate_response(q)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "response_type": response_type
            })
            st.session_state.scroll_to_bottom = True
            st.session_state.scroll_count += 1
            st.rerun()
    
    st.divider()
    st.caption(f"📁 로드된 제도: {len(PROGRAM_INFO)}개")
    st.caption(f"📁 로드된 FAQ: {len(FAQ_DATA)}개")
    
    if st.session_state.is_admin:
        st.caption(f"🔍 로드된 키워드: {len(KEYWORDS_DATA)}개")
        st.caption(f"💬 피드백 수: {len(st.session_state.feedback_data)}개")

# === 메인 콘텐츠 ===
st.title("🎓 유연학사제도(다전공) 안내 챗봇")

if menu == "💬 챗봇":
    # --- 상단 질문 예시 가이드 ---
    st.markdown("### 💡 이런 식으로 질문해 보세요!")

    # 추천 질문 리스트 (AI의 강점을 보여줄 수 있는 질문들)
    example_questions = [
        "행정학전공 2학년 과목 추천해줘",
        "복수전공과 부전공의 차이점은?",
        "융합전공에는 어떤 전공이 있어?", # AI가 리스트를 긁어오도록 유도
        "다전공 신청 기간과 방법 알려줘",
        "경영학전공 사무실 연락처랑 위치 어디야?", # 구체적인 예시로 변경
        "복수전공 신청 시 졸업 이수 학점 변화는?"
    ]

    cols = st.columns(3)    
    for idx, question in enumerate(example_questions):
        with cols[idx % 3]:
            # 말풍선 모양처럼 보이도록 스타일링된 버튼
            if st.button(f"💬 {question}", key=f"ex_q_{idx}", use_container_width=True):
                # 버튼 클릭 시 해당 질문을 채팅창에 입력한 것과 동일하게 작동
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.spinner("AI 상담원이 답변을 준비 중입니다..."):
                    try:
                        ai_response, res_type = generate_ai_response(question, st.session_state.chat_history[:-1])
                        if res_type == "error":
                            raise Exception(ai_response)
                    except Exception as e:
                        # AI 실패 시 기존 검색 로직으로 작동
                        ai_response, res_type = generate_response(question)
                        ai_response = f"⚠️ (AI 모드 일시 오류: {str(e)[:30]})\n\n" + ai_response

                st.session_state.chat_history.append({
                    "role": "assistant", 
                    "content": ai_response,
                    "response_type": res_type
                })
                st.rerun()

    st.divider()
    
    # 졸업학점 계산기 버튼 추가
    col_calc1, col_calc2 = st.columns([3, 1])
    with col_calc1:
        st.write("**💡 나에게 맞는 다전공을 찾고 싶다면?**")
    with col_calc2:
        if st.button("🧮 졸업학점 계산하기", type="primary", use_container_width=True):
            st.session_state.show_calculator = not st.session_state.show_calculator
            st.rerun()
    
    # 계산기 폼 표시
    if st.session_state.show_calculator:
        with st.container():
            st.markdown("---")
            st.subheader("📝 기본 정보 입력")
            st.write("현재 상태를 입력하면 맞춤형 다전공을 추천해드립니다!")
            
            with st.form("credit_calculator_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    current_year = datetime.now().year
                    admission_year = st.number_input(
                        "입학연도 (학번)", 
                        min_value=2020, 
                        max_value=current_year, 
                        value=current_year,
                        help="본인의 입학연도를 입력하세요"
                    )
                    
                    all_majors = sorted(PRIMARY_REQUIREMENTS['전공명'].unique().tolist()) if not PRIMARY_REQUIREMENTS.empty else []
                    primary_major = st.selectbox(
                        "본전공 (제1전공)",
                        all_majors if all_majors else ["전공 정보 없음"],
                        help="현재 본인의 본전공을 선택하세요"
                    )
                
                with col2:
                    current_grade = st.selectbox(
                        "현재 학년",
                        [1, 2, 3, 4],
                        index=1,
                        help="현재 재학 중인 학년"
                    )
                    
                    current_semester = st.radio(
                        "현재 학기",
                        [1, 2],
                        horizontal=True,
                        help="1학기 또는 2학기"
                    )
                
                col3, col4 = st.columns(2)
                
                with col3:
                    completed_required = st.number_input(
                        "이수한 전공필수 학점",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=3,
                        help="현재까지 이수한 본전공 필수 학점"
                    )
                
                with col4:
                    completed_elective = st.number_input(
                        "이수한 전공선택 학점",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=3,
                        help="현재까지 이수한 본전공 선택 학점"
                    )
                
                submitted = st.form_submit_button("🎯 다전공 추천 받기", use_container_width=True)
                
                if submitted:
                    if not all_majors or primary_major == "전공 정보 없음":
                        st.error("❌ 본전공 데이터가 없습니다. 관리자에게 문의하세요.")
                    else:
                        # 사용자 질문 추가
                        user_query = f"[졸업학점 계산 요청]\n학번: {admission_year}, 전공: {primary_major}, {current_grade}학년 {current_semester}학기\n전필: {completed_required}학점, 전선: {completed_elective}학점"
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": user_query
                        })
                        
                        # 분석 수행
                        primary_result = calculate_remaining_credits(
                            primary_major, 
                            admission_year, 
                            completed_required, 
                            completed_elective
                        )
                        
                        if primary_result is None:
                            response = f"❌ {primary_major}의 졸업요건 데이터를 찾을 수 없습니다."
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "response_type": "calculation_error"
                            })
                        else:
                            # 추천 수행
                            recommendations = recommend_programs(
                                primary_major,
                                admission_year,
                                current_grade,
                                completed_required,
                                completed_elective
                            )
                            
                            remaining_semesters = 8 - (current_grade * 2 - (2 - current_semester))
                            
                            # 결과 메시지 생성
                            response = f"""
## 📊 현재 상태 분석

- **이수 진행률:** {primary_result['progress']:.1f}%
- **이수 완료:** {primary_result['completed_total']}학점 / {primary_result['total_required']}학점
- **남은 학점:** {primary_result['total_remaining']}학점
- **남은 학기:** {remaining_semesters}학기

**상세 정보:**
- 전공필수: {primary_result['required_credits']}학점 (남은: {primary_result['remaining_required']}학점)
- 전공선택: {primary_result['elective_credits']}학점 (남은: {primary_result['remaining_elective']}학점)

---

## 🎓 맞춤형 다전공 추천
"""
                            
                            if not recommendations:
                                response += "\n추천할 수 있는 제도가 없습니다."
                            else:
                                # 가능성별로 그룹화
                                high_rec = [r for r in recommendations if r['feasibility'] == "높음"]
                                medium_rec = [r for r in recommendations if r['feasibility'] == "보통"]
                                low_rec = [r for r in recommendations if r['feasibility'] in ["낮음", "매우 낮음"]]
                                
                                # 높은 가능성
                                if high_rec:
                                    response += "\n### 🟢 추천 (높은 가능성)\n\n"
                                    for idx, rec in enumerate(high_rec[:3], 1):
                                        response += f"**{idx}. {rec['program']}** (난이도: {'★' * rec['difficulty']}{'☆' * (5 - rec['difficulty'])})\n"
                                        response += f"- 필요 학점: {rec['required_credits']}학점"
                                        if rec['additional_primary_credits'] > 0:
                                            response += f" (본전공 +{rec['additional_primary_credits']}학점)"
                                        response += f"\n- 여유 학점: {rec['net_available']}학점\n"
                                        response += f"- 판단 이유:\n"
                                        for reason in rec['reasons']:
                                            response += f"  {reason}\n"
                                        response += f"\n**액션 플랜:**\n"
                                        action_plan = generate_action_plan(rec, current_grade, remaining_semesters)
                                        response += action_plan + "\n\n"
                                
                                # 보통
                                if medium_rec:
                                    response += "\n### 🟡 고려 가능 (보통)\n\n"
                                    for idx, rec in enumerate(medium_rec[:2], 1):
                                        response += f"**{idx}. {rec['program']}** (난이도: {'★' * rec['difficulty']}{'☆' * (5 - rec['difficulty'])})\n"
                                        response += f"- 필요 학점: {rec['required_credits']}학점"
                                        if rec['additional_primary_credits'] > 0:
                                            response += f" (본전공 +{rec['additional_primary_credits']}학점)"
                                        response += f"\n- 여유 학점: {rec['net_available']}학점\n"
                                        response += f"- 판단 이유:\n"
                                        for reason in rec['reasons']:
                                            response += f"  {reason}\n"
                                        response += "\n"
                                
                                # 낮음
                                if low_rec and not high_rec and not medium_rec:
                                    response += "\n### 🔴 신중 검토 필요 (낮음)\n\n"
                                    for idx, rec in enumerate(low_rec[:2], 1):
                                        response += f"**{idx}. {rec['program']}**\n"
                                        response += f"- 필요 학점: {rec['total_needed']}학점, 여유: {rec['net_available']}학점\n"
                                        for reason in rec['reasons']:
                                            response += f"  {reason}\n"
                                        response += "\n"
                                
                                # 종합 조언
                                response += "\n---\n\n## 💬 종합 조언\n\n"
                                
                                if high_rec:
                                    response += f"""
**🎉 좋은 소식입니다!**

현재 상태에서 {len(high_rec)}개의 제도를 무리 없이 이수할 수 있습니다.

**다음 단계:**
1. 관심 있는 다전공 제도 확인 ('📚 다전공  제도 안내' 메뉴)
2. 해당 전공 사무실 또는 학사지원팀 상담(챗봇에서 전공 검색)
3. 다전공제도 신청 기간(학기별)에 맞춰 신청서 제출
"""
                                elif medium_rec:
                                    response += f"""
**🤔 신중한 계획이 필요합니다**

{len(medium_rec)}개의 제도가 가능하지만, 학기당 이수 학점을 높여야 합니다.
**{medium_rec[0]['program']}**을(를) 고려해보세요.

**권장 사항:**
1. 본전공 학점 이수 계획 먼저 확정
2. 계절학기 활용 계획 수립
3. 학사지원팀에서 정확한 이수 가능성 확인
"""
                                else:
                                    response += """
**⚠️ 현실적인 대안을 고려하세요**

현재 상태에서는 학점 부담이 높은 제도보다는
**부전공(21학점)** 또는 **마이크로디그리(12~18학점)**를 추천드립니다.

**대안:**
1. 낮은 학점의 제도 선택
2. 계절학기 적극 활용
3. 졸업 후 추가 학기 고려
4. 전문가 상담 필수
"""
                                
                                response += "\n\n📞 **문의:** 학사지원팀 (031-670-5035)"
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "response_type": "calculation_result"
                            })
                        
                        st.session_state.show_calculator = False
                        st.session_state.scroll_to_bottom = True
                        st.session_state.scroll_count += 1
                        st.rerun()
            
            st.markdown("---")
    
    st.divider()
    
    # 채팅 히스토리 표시
    for idx, chat in enumerate(st.session_state.chat_history):
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.write(chat["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(chat["content"])
                
                # 피드백 버튼
                if chat.get("response_type") in ["faq", "program", "comparison", "credits", "application", "graduation", "calculation_result"]:
                    feedback_key = f"feedback_{idx}"
                    
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
                    
                    elif st.session_state.show_feedback[feedback_key] == "helpful":
                        st.success("✅ 피드백 감사합니다!")
                    elif st.session_state.show_feedback[feedback_key] == "not_helpful":
                        st.info("📝 피드백 감사합니다. 더 나은 답변을 위해 노력하겠습니다!")

    # 사용자 입력
    # 챗봇 입력창 부분
user_input = st.chat_input("메시지를 입력하세요...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("AI가 응답을 생성 중입니다..."):
        # 1. AI 응답을 시도합니다.
        ai_response, res_type = generate_ai_response(user_input, st.session_state.chat_history[:-1])
        
        # [중요] 만약 에러가 발생했다면, 화면에 빨간색으로 에러를 다 보여줍니다.
        if res_type == "error":
            st.error(f"❌ AI가 작동하지 않는 진짜 이유: {ai_response}")
            # 여기서 멈춥니다. 아래 fallback 로직으로 넘어가지 않게 합니다.
            st.stop() 

    # 에러가 없을 때만 정상적으로 채팅 기록에 추가합니다.
    st.session_state.chat_history.append({
        "role": "assistant", 
        "content": ai_response,
        "response_type": res_type
    })
    st.rerun()

    # 스크롤 로직
    if st.session_state.scroll_to_bottom:
        scroll_to_bottom()
        st.session_state.scroll_to_bottom = False

elif menu == "📚 다전공 제도 안내":
    
    st.header("📊 제도 한눈에 비교")

    # 3열 그리드 생성
    cols = st.columns(3)

    for idx, (program, info) in enumerate(PROGRAM_INFO.items()):
        with cols[idx % 3]:
            # 데이터 가져오기
            desc = info.get('description', '설명 없음')
            c_pri = info.get('credits_primary', '-')
            c_mul = info.get('credits_multi', '-')
            
            # 졸업인증/시험 여부
            cert_val = str(info.get('graduation_certification', '-'))
            exam_val = str(info.get('graduation_exam', '-'))
            grad_cert = info.get('graduation_certification', '-')
            grad_exam = info.get('graduation_exam', '-')
            
            degree = info.get('degree', '-')
            difficulty = info.get('difficulty', '⭐')

            # 스타일 정의 (한 줄로 유지)
            long_text_style = "overflow: hidden; text-overflow: ellipsis; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; line-height: 1.4;"

            # [핵심 수정] HTML 코드를 왼쪽 벽(시작점)에 딱 붙여서 작성했습니다.
            # 이렇게 해야 마크다운이 '코드 블록'으로 오해하지 않고 정상적으로 렌더링합니다.
            html_content = f"""
<div style="border: 1px solid #e5e7eb; border-radius: 14px; padding: 18px; background: white; box-shadow: 0 4px 6px rgba(0,0,0,0.05); min-height: 380px; margin-bottom: 20px; display: flex; flex-direction: column; justify-content: space-between;">
    <div>
        <h3 style="margin: 0 0 8px 0; color: #1f2937; font-size: 1.2rem;">🎓 {program}</h3>
        <p style="color: #6b7280; font-size: 14px; margin-bottom: 12px; {long_text_style}">{desc}</p>
        <hr style="margin: 12px 0; border: 0; border-top: 1px solid #e5e7eb;">
        <div style="font-size: 14px; margin-bottom: 8px;">
            <strong style="color: #374151;">📖 이수 학점</strong>
            <ul style="padding-left: 18px; margin: 4px 0; color: #4b5563;">
                <li style="margin-bottom: 4px; {long_text_style}"><span style="font-weight:600; color:#374151;">본전공:</span> {c_pri}</li>
                <li style="{long_text_style}"><span style="font-weight:600; color:#374151;">다전공:</span> {c_mul}</li>
            </ul>
        </div>
        <div style="font-size: 14px; margin-bottom: 12px;">
            <strong style="color: #374151;">🎓 졸업 요건</strong>
            <ul style="padding-left: 18px; margin: 4px 0; color: #4b5563;">
                <li>졸업인증: {grad_cert}</li>
                <li>졸업시험: {grad_exam}</li>
            </ul>
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: end; margin-top: 10px;">
        <div style="max-width: 65%;">
            <strong style="color: #374151; font-size: 14px;">📜 학위</strong><br>
            <div style="font-size: 13px; color: #2563eb; background: #eff6ff; padding: 2px 6px; border-radius: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{degree}</div>
        </div>
        <div style="text-align: right; min-width: 30%;">
            <strong style="color: #374151; font-size: 14px;">난이도</strong><br>
            <span style="color: #f59e0b; font-size: 16px;">{difficulty}</span>
        </div>
    </div>
</div>"""

            st.markdown(html_content, unsafe_allow_html=True)

    st.divider()
    
    # === 2. 상세 정보 보기 (기존 기능 복원 및 통합) ===
    st.header("🔍 상세 제도 안내")
    
    selected_program = st.selectbox("자세히 알아볼 제도를 선택하세요", list(PROGRAM_INFO.keys()))
    
    if selected_program:
        info = PROGRAM_INFO[selected_program]
        
        # 탭을 사용하여 정보 구조화
        tab1, tab2 = st.tabs(["📝 기본 정보", "✅ 특징 및 유의사항"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("개요")
                st.write(info.get('description', ''))
                
                st.subheader("이수 학점 상세")
                st.markdown(f"""
                - **교양 필수:** {info.get('credits_general', '-')}
                - **원전공 필수:** {info.get('credits_primary', '-')}
                - **다전공 필수:** {info.get('credits_multi', '-')}
                """)
                
                st.subheader("졸업 요건")
                st.markdown(f"""
                - **졸업인증:** {info.get('graduation_certification', '-')}
                - **졸업시험:** {info.get('graduation_exam', '-')}
                """)
                
            with col2:
                st.info(f"**신청 자격**\n\n{info.get('qualification', '-')}")
                st.success(f"**학위 표기**\n\n{info.get('degree', '-')}")
                st.metric(f"**✨ 난이도**", info['difficulty'])


        with tab2:
            st.subheader("특징")
            features = info.get('features', [])
            if features and isinstance(features, list) and len(features) > 0 and features[0] != '':
                for f in features:
                    st.write(f"✔️ {f.strip()}")
            else:
                st.write("등록된 특징이 없습니다.")
            
            if info.get('notes'):
                st.warning(f"**💡 기타 유의사항:**\n{info['notes']}")

    st.divider()
      
    
    # 전공 목록 가져오기
    available_majors = set()
    
    if not COURSES_DATA.empty:
        majors_in_courses = COURSES_DATA[
            COURSES_DATA['제도유형'] == selected_program
        ]['전공명'].unique().tolist()
        available_majors.update(majors_in_courses)
        
    if not CURRICULUM_MAPPING.empty:
        majors_in_mapping = CURRICULUM_MAPPING[
            CURRICULUM_MAPPING['제도유형'] == selected_program
        ]['전공명'].unique().tolist()
        available_majors.update(majors_in_mapping)
    
    # 전공 선택 및 정보 표시
    if available_majors:
        target_programs = ["복수전공", "부전공", "융합전공", "융합부전공"]
        
        if selected_program in target_programs:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors)))
            with col_m2:
                all_majors = sorted(PRIMARY_REQUIREMENTS['전공명'].unique().tolist()) if not PRIMARY_REQUIREMENTS.empty else []
                my_primary_major = st.selectbox("나의 본전공 (제1전공)", ["선택 안 함"] + all_majors)
        else:
            selected_major = st.selectbox(f"이수하려는 {selected_program}", sorted(list(available_majors)))
            my_primary_major = "선택 안 함"

        # 학점 요건 조회
        if selected_program in target_programs:
            current_year = datetime.now().year
            admission_year = st.number_input(
                "본인 학번 (입학연도)", 
                min_value=2020, 
                max_value=current_year, 
                value=current_year
            )
            
            st.write("")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader(f"🎯 {selected_program}({selected_major}) 이수 학점 기준")
                
                if not GRAD_REQUIREMENTS.empty:
                    req_data = GRAD_REQUIREMENTS[
                        (GRAD_REQUIREMENTS['전공명'] == selected_major) & 
                        (GRAD_REQUIREMENTS['제도유형'] == selected_program)
                    ].copy()
                    
                    req_data['기준학번'] = pd.to_numeric(req_data['기준학번'], errors='coerce')
                    req_data = req_data.dropna(subset=['기준학번'])
                    applicable = req_data[req_data['기준학번'] <= admission_year]
                    
                    if not applicable.empty:
                        applicable = applicable.sort_values('기준학번', ascending=False)
                        row = applicable.iloc[0]
                        
                        st.write(f"- 전공필수: **{int(row['전공필수'])}**학점")
                        st.write(f"- 전공선택: **{int(row['전공선택'])}**학점")
                        st.markdown(f"#### 👉 {selected_program} {int(row['총학점'])}학점 이수")
                    else:
                        st.warning(f"{admission_year}학번 기준 데이터가 없습니다.")
                else:
                    st.warning("졸업요건 파일이 없습니다.")

            with col_right:
                st.subheader(f"🏠 본전공({my_primary_major}) 이수 학점 기준")
                
                if my_primary_major != "선택 안 함" and not PRIMARY_REQUIREMENTS.empty:
                    pri_data = PRIMARY_REQUIREMENTS[PRIMARY_REQUIREMENTS['전공명'] == my_primary_major].copy()
                    pri_data['기준학번'] = pd.to_numeric(pri_data['기준학번'], errors='coerce')
                    pri_valid = pri_data[pri_data['기준학번'] <= admission_year]
                    
                    if not pri_valid.empty:
                        matched_row = None
                        pri_valid = pri_valid.sort_values('기준학번', ascending=False)
                        
                        for _, p_row in pri_valid.iterrows():
                            if selected_program in str(p_row['구분']):
                                matched_row = p_row
                                break
                        
                        if matched_row is not None:
                            st.write(f"- 본전공 전필: **{int(matched_row['본전공_전필'])}**학점")
                            st.write(f"- 본전공 전선: **{int(matched_row['본전공_전선'])}**학점")
                            st.markdown(f"#### 👉 본전공 {int(matched_row['본전공_계'])}학점 이수")
                            
                            if pd.notna(matched_row.get('비고')):
                                st.caption(f"참고: {matched_row['비고']}")
                        else:
                            st.info(f"변동 데이터가 없습니다.")
                    else:
                        st.warning(f"{admission_year}학번 기준 데이터가 없습니다.")
                elif my_primary_major == "선택 안 함":
                    st.info("본전공을 선택하면 변동된 이수 학점을 확인할 수 있습니다.")

        st.divider()

        # 이미지 표시
        if selected_program == "융합전공" or "소단위전공" in selected_program:
            title = "📋 이수체계도" if selected_program == "융합전공" else "🖼️ 과정 안내 이미지"
            st.subheader(title)
            display_curriculum_image(selected_major, selected_program)
        
        # 이수 과목 표시
        if not COURSES_DATA.empty:
            display_courses(selected_major, selected_program)

elif menu == "❓ FAQ":
    st.header("자주 묻는 질문 (FAQ)")
    
    categories = list(set([faq["카테고리"] for faq in FAQ_DATA]))
    selected_category = st.selectbox("카테고리 선택", ["전체"] + categories)
    
    filtered_faqs = FAQ_DATA if selected_category == "전체" else [faq for faq in FAQ_DATA if faq["카테고리"] == selected_category]
    
    for i, faq in enumerate(filtered_faqs):
        with st.expander(f"Q. {faq['질문']}"):
            st.write(f"**A.** {faq['답변']}")
            st.caption(f"카테고리: {faq['카테고리']}")
            

elif menu == "🔑 키워드 관리":
    st.header("키워드 관리 (관리자 전용)")
    st.write("등록된 키워드를 확인하고 검색 테스트를 해보세요.")
    
    st.subheader("🔍 키워드 검색 테스트")
    test_input = st.text_input("테스트할 문장을 입력하세요", placeholder="예: 복전 학점은?")
    
    if test_input:
        keyword_match = search_by_keyword(test_input)
        if keyword_match:
            st.success(f"✅ 키워드 매칭 성공!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("매칭된 키워드", keyword_match['키워드'])
            with col2:
                st.metric("타입", keyword_match['타입'])
            with col3:
                st.metric("연결정보", keyword_match['연결정보'])
        else:
            st.warning("❌ 매칭되는 키워드가 없습니다. 유사도 검색으로 진행됩니다.")
    
    st.divider()
    
    st.subheader("📋 등록된 키워드 목록")
    
    if KEYWORDS_DATA:
        keyword_types = list(set([k['타입'] for k in KEYWORDS_DATA]))
        selected_type = st.selectbox("타입 필터", ["전체"] + keyword_types)
        
        if selected_type == "전체":
            filtered_keywords = KEYWORDS_DATA
        else:
            filtered_keywords = [k for k in KEYWORDS_DATA if k['타입'] == selected_type]
        
        keyword_df = pd.DataFrame(filtered_keywords)
        st.dataframe(
            keyword_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "키워드": st.column_config.TextColumn("키워드", width="medium"),
                "타입": st.column_config.TextColumn("타입", width="small"),
                "연결정보": st.column_config.TextColumn("연결정보", width="medium")
            }
        )
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 키워드 수", len(KEYWORDS_DATA))
        with col2:
            program_keywords = [k for k in KEYWORDS_DATA if k['타입'] == '제도']
            st.metric("제도 키워드", len(program_keywords))
        with col3:
            topic_keywords = [k for k in KEYWORDS_DATA if k['타입'] == '주제']
            st.metric("주제 키워드", len(topic_keywords))
        
        st.info("""
💡 **키워드 추가 방법**
1. `data/keywords.xlsx` 파일 열기
2. 새로운 행 추가 (키워드, 타입, 연결정보)
3. 파일 저장 후 앱 새로고침

**타입 종류:**
- `제도`: 특정 제도로 연결
- `주제`: 주제별 정보로 연결
        """)
    else:
        st.warning("등록된 키워드가 없습니다.")

elif menu == "📊 피드백 통계":
    st.header("피드백 통계 (관리자 전용)")
    
    if st.session_state.feedback_data:
        feedback_df = pd.DataFrame(st.session_state.feedback_data)
        
        st.subheader("📈 전체 통계")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            helpful_count = len(feedback_df[feedback_df['feedback'] == 'helpful'])
            st.metric("👍 도움됨", helpful_count)
        
        with col2:
            not_helpful_count = len(feedback_df[feedback_df['feedback'] == 'not_helpful'])
            st.metric("👎 아님", not_helpful_count)
        
        with col3:
            total = len(feedback_df)
            satisfaction = (helpful_count / total * 100) if total > 0 else 0
            st.metric("만족도", f"{satisfaction:.1f}%")
        
        st.divider()
        
        st.subheader("📋 최근 피드백")
        
        feedback_filter = st.selectbox(
            "피드백 타입",
            ["전체", "도움됨", "아님"]
        )
        
        if feedback_filter == "도움됨":
            filtered_feedback = feedback_df[feedback_df['feedback'] == 'helpful']
        elif feedback_filter == "아님":
            filtered_feedback = feedback_df[feedback_df['feedback'] == 'not_helpful']
        else:
            filtered_feedback = feedback_df
        
        filtered_feedback = filtered_feedback.sort_values('timestamp', ascending=False)
        
        st.dataframe(
            filtered_feedback[['question', 'feedback', 'timestamp']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("질문", width="large"),
                "feedback": st.column_config.TextColumn("피드백", width="small"),
                "timestamp": st.column_config.DatetimeColumn(
                    "시간",
                    format="YYYY-MM-DD HH:mm"
                )
            }
        )
    else:
        st.info("아직 수집된 피드백이 없습니다.")

st.caption("💡 더 자세한 정보는 학사지원팀(031-670-5035) 또는 전공 사무실에 문의하세요.")
st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y년 %m월 %d일')}")
