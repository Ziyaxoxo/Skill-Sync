import re
import PyPDF2
import pickle
import random

# Load models if they exist
try:
    clf = pickle.load(open('rf_classifier.pkl', 'rb'))
    vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    MODELS_LOADED = True
except:
    MODELS_LOADED = False

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        return str(e)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_category(resume_text):
    if not MODELS_LOADED:
        return "General"
    cleaned_text = clean_text(resume_text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = clf.predict(vectorized_text)[0]
    return prediction

def extract_skills(text):
    # EXPANDED SKILL LIST
    skills_db = [
        # Languages
        'python', 'java', 'c++', 'javascript', 'typescript', 'c#', 'go', 'ruby', 'php', 'swift', 'kotlin', 'rust',
        # Frontend
        'html', 'css', 'react', 'angular', 'vue', 'redux', 'tailwind', 'bootstrap', 'jquery',
        # Backend
        'node', 'express', 'django', 'flask', 'spring boot', 'dotnet', 'rails', 'fastapi',
        # Database
        'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'firebase', 'cassandra',
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab', 'terraform', 'ansible', 'circleci',
        # Data Science & ML
        'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        # Tools & Concepts
        'agile', 'scrum', 'jira', 'tableau', 'power bi', 'excel', 'linux', 'bash', 'rest api', 'graphql', 'system design', 'microservices'
    ]
    
    found_skills = set()
    for skill in skills_db:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            found_skills.add(skill)
    return list(found_skills)

def calculate_ats_score(resume_text, missing_skills, jd_text):
    """
    Calculates a simulated ATS score based on common parsing rules.
    """
    score = 0
    breakdown = []
    
    # 1. KEYWORD MATCHING (50 points)
    total_jd_keywords = len(set(extract_skills(jd_text)))
    if total_jd_keywords > 0:
        missing_count = len(missing_skills)
        match_ratio = (total_jd_keywords - missing_count) / total_jd_keywords
        keyword_score = round(match_ratio * 50, 1)
    else:
        keyword_score = 50
    score += keyword_score
    breakdown.append(f"• **Keywords:** {keyword_score}/50 points")

    # 2. CONTACT INFO CHECK (20 points)
    contact_score = 0
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    if re.search(email_pattern, resume_text):
        contact_score += 10
        breakdown.append("• **Email:** Found (+10 pts)")
    else:
        breakdown.append("• **Email:** ❌ Not detected")

    phone_pattern = r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
    if re.search(phone_pattern, resume_text):
        contact_score += 10
        breakdown.append("• **Phone:** Found (+10 pts)")
    else:
        breakdown.append("• **Phone:** ❌ Not detected")
    score += contact_score

    # 3. SECTION HEADERS CHECK (20 points)
    section_score = 0
    required_sections = ["experience", "education", "skills", "projects"]
    found_sections = []
    lower_text = resume_text.lower()
    for sec in required_sections:
        if sec in lower_text:
            found_sections.append(sec.title())
            section_score += 5
    score += section_score
    missing_sections = set([s.title() for s in required_sections]) - set(found_sections)
    if len(found_sections) == 4:
        breakdown.append(f"• **Sections:** All essential sections found (+20 pts)")
    else:
        breakdown.append(f"• **Sections:** Found {len(found_sections)}/4. Missing: {', '.join(missing_sections)}")

    # 4. LENGTH CHECK (10 points)
    word_count = len(resume_text.split())
    length_score = 0
    if 300 <= word_count <= 1200:
        length_score = 10
        breakdown.append(f"• **Length:** Optimal ({word_count} words) (+10 pts)")
    else:
        breakdown.append(f"• **Length:** ⚠️ {word_count} words (Aim for 300-1200)")
    score += length_score
    
    return round(score), breakdown

def get_static_interview_prep(missing_skills):
    """
    Returns interview questions from a MASSIVE database based on missing skills.
    """
    
    # --- TECHNICAL QUESTION BANK ---
    question_bank = {
        # Languages
        "python": "Explain the difference between deep copy and shallow copy. What are decorators?",
        "java": "Explain the difference between JDK, JRE, and JVM. How does Garbage Collection work?",
        "c++": "What are virtual functions? Explain the difference between pointers and references.",
        "javascript": "Explain Closures and Hoisting. What is the difference between '==' and '==='?",
        "typescript": "What are Interfaces vs Types? How do you handle generics in TypeScript?",
        "c#": "What is the difference between ref and out parameters? Explain Boxing and Unboxing.",
        "go": "What are Goroutines? Explain the difference between arrays and slices.",
        "ruby": "What is a Gem? Explain the difference between Proc and Lambda.",
        "php": "What are the superglobal variables in PHP? Explain strict types.",
        "swift": "What are Optionals? Explain the difference between struct and class in Swift.",

        # Frontend
        "html": "What are semantic tags? Explain the difference between localStorage, sessionStorage, and cookies.",
        "css": "Explain the Box Model. What is the difference between Flexbox and Grid?",
        "react": "What are the rules of Hooks? Explain the useEffect dependency array and React Fiber.",
        "angular": "What is Dependency Injection? Explain the difference between Observables and Promises.",
        "vue": "Explain the Vue lifecycle. What is the difference between v-show and v-if?",
        "redux": "Explain the Redux data flow. What are Actions and Reducers?",
        "tailwind": "What are the benefits of utility-first CSS? How do you configure a custom theme?",

        # Backend
        "node": "Explain the Event Loop. What is the difference between process.nextTick() and setImmediate()?",
        "express": "What is Middleware in Express? How do you handle error handling globally?",
        "django": "Explain the MVT architecture. What is the purpose of migrations?",
        "flask": "What is a Blueprint in Flask? How do you handle request contexts?",
        "spring boot": "What is Auto-configuration? Explain the @SpringBootApplication annotation.",
        "rest api": "What are the idempotent HTTP methods? Explain status codes 401 vs 403.",
        "graphql": "What is the difference between Query and Mutation? How do you solve the N+1 problem?",

        # Database
        "sql": "Write a query to find duplicates in a table. Explain Indexing and Normalization.",
        "postgresql": "What is MVCC? Explain the difference between JSON and JSONB types.",
        "mongodb": "What is the Aggregation Framework? Explain Sharding vs Replication.",
        "redis": "What are the common data types in Redis? How is it used for Caching?",

        # Cloud & DevOps
        "aws": "Explain the difference between S3, EBS, and EFS. What is a VPC?",
        "azure": "What is an Azure Resource Manager template? Explain Blob Storage.",
        "docker": "Explain the difference between ENTRYPOINT and CMD. What is a Docker Volume?",
        "kubernetes": "What is a Pod? Explain the difference between a Deployment and a StatefulSet.",
        "jenkins": "How do you create a Multibranch Pipeline? What are Jenkins shared libraries?",
        "git": "Explain 'git rebase' vs 'git merge'. How do you resolve a merge conflict?",
        "terraform": "What is State in Terraform? Explain 'terraform plan' vs 'terraform apply'.",

        # Data & ML
        "machine learning": "Explain the Bias-Variance tradeoff. What is Cross-Validation?",
        "deep learning": "What is Backpropagation? Explain the Vanishing Gradient problem.",
        "nlp": "What is Tokenization? Explain Word Embeddings (Word2Vec/GloVe).",
        "pandas": "How do you handle missing data? Explain the difference between loc and iloc.",
        "tensorflow": "What are Tensors? Explain the difference between Sequential and Functional APIs.",
        "scikit-learn": "What is a Pipeline? How do you perform hyperparameter tuning using GridSearch?",

        # General / Tools
        "system design": "Design a URL shortener (like Bit.ly). How would you handle scaling?",
        "agile": "What are the ceremonies in Scrum? Explain the difference between Kanban and Scrum.",
        "microservices": "What are the advantages of Microservices? How do services communicate?",
        "linux": "What is the difference between 'grep', 'awk', and 'sed'? Check process usage with 'top'.",
        "excel": "Explain VLOOKUP vs INDEX-MATCH. How do you create a Pivot Table?"
    }

    advice = "### 🎓 Customized Interview Prep\n\n"
    
    # 1. TECHNICAL QUESTIONS
    if missing_skills:
        advice += "**1. Targeted Technical Questions (Based on your gaps):**\n"
        count = 0
        for skill in missing_skills:
            if skill in question_bank:
                advice += f"- **{skill.title()}:** {question_bank[skill]}\n"
                count += 1
            if count >= 5: 
                break
        
        if count == 0:
             advice += "- **General:** Since your gaps are niche, focus on the fundamentals of the job description's core domain.\n"
    else:
        advice += "**1. Technical Questions:**\n- Your skills match perfectly! Expect advanced system design, architecture, or behavioral questions.\n"

    # --- 2. EXTENSIVE BEHAVIORAL BANK (20 Questions) ---
    behavioral_bank = [
        "Tell me about a time you had a conflict with a coworker. How did you resolve it?",
        "Describe a situation where you had to meet a tight deadline. How did you prioritize?",
        "Tell me about a time you failed. What did you learn from it?",
        "Describe a complex problem you solved. What was your thought process?",
        "How do you handle constructive criticism?",
        "Tell me about a time you showed leadership skills.",
        "Why do you want to work for this specific role/industry?",
        "Describe a time you had to learn a new technology quickly.",
        "Tell me about a time you disagreed with a supervisor's decision.",
        "Describe a time you went above and beyond for a project.",
        "How do you handle working with a difficult client or stakeholder?",
        "Tell me about a mistake you made. How did you fix it?",
        "Describe a time you had to persuade others to your way of thinking.",
        "How do you stay organized when you have multiple projects?",
        "Tell me about a time you had to adapt to a significant change at work.",
        "Describe a time you mentored a junior team member.",
        "What is your proudest professional achievement?",
        "Tell me about a time you identified a process inefficiency and fixed it.",
        "How do you maintain motivation during repetitive tasks?",
        "Describe a time you had to deliver bad news to a team or client."
    ]
    # Pick 3 random behavioral questions
    selected_behavioral = random.sample(behavioral_bank, 3)
    
    advice += "\n**2. Behavioral Questions (Practice these):**\n"
    for q in selected_behavioral:
        advice += f"- {q}\n"

    # --- 3. EXTENSIVE STRATEGIC TIPS BANK (20 Tips) ---
    strategic_tips_bank = [
        "**The 'So What?' Test:** For every answer, explain the impact. Don't just say what you did; say why it mattered to the business.",
        "**Body Language:** Maintain eye contact (even on Zoom, look at the camera). Keep your hands visible to build trust.",
        "**The Reverse Interview:** Ask them: 'What is the biggest challenge the team is facing right now?' It shows you care about solving problems.",
        "**STAR Method:** Always structure behavioral answers with Situation, Task, Action, and Result.",
        "**Research Competitors:** Mentioning a competitor's recent move shows you understand the market landscape.",
        "**Silence is Okay:** It's better to pause for 5 seconds to think than to ramble for 2 minutes.",
        "**Quantify Results:** Use numbers wherever possible (e.g., 'Improved load time by 20%', 'Managed a budget of $50k').",
        "**The 'Weakness' Question:** Choose a real weakness but explain the specific steps you are taking to improve it.",
        "**Cultural Fit:** specific Use keywords from their 'About Us' page (e.g., 'Innovation', 'Customer Obsession') in your answers.",
        "**First 5 Minutes:** The impression is often made in the intro. Have a polished 'Tell me about yourself' pitch ready.",
        "**Technical Clarity:** If you don't know a technical answer, explain how you would find out (e.g., 'I would check the documentation for X').",
        "**Post-Interview Note:** Send a thank-you email within 24 hours referencing a specific topic you discussed.",
        "**Mock Interviews:** Record yourself answering common questions to catch filler words like 'um' and 'like'.",
        "**Salary Negotiation:** Don't give a number first. Ask for the budget range for the role.",
        "**LinkedIn Alignment:** Ensure your resume dates and titles match your LinkedIn profile exactly.",
        "**Github Readme:** If sharing code, ensure your repositories have a README explaining what the project does and how to run it.",
        "**Soft Skills:** Highlight communication and teamwork, not just coding. Engineering is a team sport.",
        "**Ask About Success:** Ask 'What does success look like in this role for the first 90 days?'",
        "**Handling Stress:** Be ready to explain your personal strategies for managing burnout and tight deadlines.",
        "**Continuous Learning:** Mention a podcast, book, or course you are currently consuming to show you stay updated."
    ]
    
    # Pick 3 random strategic tips
    selected_tips = random.sample(strategic_tips_bank, 3)

    advice += "\n**3. Strategic Tips:**\n"
    for tip in selected_tips:
        advice += f"- {tip}\n"

    # Specific Tip Logic (keeps specific advice relevant)
    if any(s in missing_skills for s in ['react', 'angular', 'vue', 'html', 'css']):
        advice += "- **Frontend Specific:** Ensure your GitHub links are working and your portfolio site is mobile-responsive.\n"
    elif any(s in missing_skills for s in ['machine learning', 'pandas', 'sql']):
        advice += "- **Data Specific:** Don't just show numbers. Explain *why* the data matters to the business decision.\n"
    elif any(s in missing_skills for s in ['aws', 'docker', 'kubernetes']):
        advice += "- **Cloud Specific:** Be ready to draw system architecture diagrams on a whiteboard.\n"

    return advice