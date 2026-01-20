import streamlit as st
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_pdf, clean_text, extract_skills, predict_category, get_static_interview_prep, calculate_ats_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Skill-Sync",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #2f325d;
        font-family: 'Inter', sans-serif;
    }
    
    /* File Uploader Background */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #404472 !important;
        border: 1px solid #4F8BF9;
    }
    
    /* Job Description Text Area */
    .stTextArea textarea {
        background-color: #404472 !important;
        color: white !important;
    }
    .stTextArea textarea:focus {
        border-color: #4F8BF9 !important;
        box-shadow: 0 0 0 1px #4F8BF9 !important;
    }

    /* Header Styling */
    h1 {
        color: white;
        font-weight: 700;
        margin-bottom: 0px;
    }
    h3, h5 {
        color: white !important;
    }
    
    /* Custom Cards for Stats */
    .stat-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 5px solid #4F8BF9;
        margin-bottom: 20px;
    }
    .stat-value {
        font-size: 32px;
        font-weight: bold;
        color: #2C3E50;
    }
    .stat-label {
        color: #7F8C8D;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Skill Tags */
    .skill-tag {
        display: inline-block;
        background-color: #E8F0FE;
        color: #1A73E8;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 14px;
        margin: 4px;
        font-weight: 500;
    }
    .missing-tag {
        background-color: #FCE8E6;
        color: #D93025;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3b74e0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("SkillSync")
    st.caption("v2.1 • Offline Mode")
    st.markdown("---")
    st.markdown("### ⚙️ How it works")
    st.info(
        "1. Upload your Resume (PDF)\n"
        "2. Paste the Job Description\n"
        "3. Get instant analysis & tips"
    )
    st.markdown("---")

# --- MAIN CONTENT ---
st.title("SkillSync")
st.markdown("##### Optimize your resume for ATS systems and get hired faster.")
st.write("") 

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown("### 💼 Job Description")
    job_description = st.text_area("Paste JD here...", height=200, label_visibility="collapsed")

if st.button("Analyze Match Compatibility"):
    if uploaded_file and job_description:
        with st.spinner("🔍 Scanning resume against job description..."):
            time.sleep(1)
            
            # --- LOGIC ---
            resume_text = extract_text_from_pdf(uploaded_file)
            clean_resume = clean_text(resume_text)
            clean_jd = clean_text(job_description)
            
            predicted_category = predict_category(clean_resume)
            
            text_corpus = [clean_resume, clean_jd]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(text_corpus)
            match_percentage = round(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100, 2)
            
            resume_skills = set(extract_skills(clean_resume))
            jd_skills = set(extract_skills(clean_jd))
            missing_skills = list(jd_skills - resume_skills)
            matching_skills = list(resume_skills.intersection(jd_skills))

            # Calculate ATS Score
            ats_score, ats_breakdown = calculate_ats_score(resume_text, missing_skills, clean_jd)

        # --- RESULTS DASHBOARD ---
        st.markdown("---")
        st.subheader("🎯 Analysis Results")
        
        # 4 Columns for Stats
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{match_percentage}%</div>
                <div class="stat-label">JD Match</div>
            </div>
            """, unsafe_allow_html=True)
        
        with m_col2:
            color = "#27AE60" if ats_score >= 80 else "#F1C40F" if ats_score >= 60 else "#E74C3C"
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: {color};">
                <div class="stat-value">{ats_score}</div>
                <div class="stat-label">ATS Score</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #27AE60;">
                <div class="stat-value">{len(matching_skills)}</div>
                <div class="stat-label">Matching Skills</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #E74C3C;">
                <div class="stat-value">{len(missing_skills)}</div>
                <div class="stat-label">Missing Skills</div>
            </div>
            """, unsafe_allow_html=True)
            
        # ATS Breakdown
        st.write("") 
        st.write("") 
        with st.expander("ℹ️ How was my ATS Score calculated?"):
            for item in ats_breakdown:
                st.write(item)

        # Progress Bar - FIXED THRESHOLDS HERE
        st.write("")
        st.write("### Compatibility Meter")
        st.progress(match_percentage / 100)
        
        # Updated Logic for "Real World" NLP Scores
        if match_percentage >= 50:
            st.success("✅ **High Match!** Your resume is very well aligned with this job.")
        elif match_percentage >= 35:
            st.warning("⚠️ **Good Match:** You have the core skills, but could tailor the content more.")
        else:
            st.error("❌ **Low Match:** Consider adding more specific keywords from the JD.")

        # Skills Analysis
        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Matching Skills")
            if matching_skills:
                for skill in matching_skills:
                    st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.info("No common skills found.")
        
        with c2:
            st.markdown("#### ⚠️ Missing Skills (Add these!)")
            if missing_skills:
                for skill in missing_skills:
                    st.markdown(f'<span class="skill-tag missing-tag">{skill}</span>', unsafe_allow_html=True)
            else:
                st.success("You have all the required skills!")

        # Interview Prep
        st.write("")
        st.write("")
        with st.expander("📚 View Smart Interview Questions (Based on your gaps)", expanded=True):
            advice = get_static_interview_prep(missing_skills)
            st.markdown(advice.replace("### 🎓 Customized Interview Prep", ""))

    else:
        st.error("Please upload a PDF resume and paste the Job Description to start.")