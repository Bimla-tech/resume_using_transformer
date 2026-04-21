"""
AI Resume Intelligence System — Strict Role-Based Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rules:
1. ROLE_SKILLS_DB  — strict dictionary of role → {core, optional} skills
2. Hybrid matching — exact keyword search + all-mpnet-base-v2 embeddings
3. Strict gap logic — missing = role_skills − resume_skills (set subtraction)
4. Weighted scoring — core skills count 2×, optional count 1×
5. Strict filter   — skills NOT in the selected role are never shown
"""

import os
import re
import io
import urllib.parse
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for
from werkzeug.utils import secure_filename

import fitz                                           # PyMuPDF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.units import inch
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = os.environ.get('SESSION_SECRET', 'resume-ai-2024')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ─────────────────────────────────────────────
# Lazy-loaded models + in-memory result cache
# ─────────────────────────────────────────────
_sentence_model = None
_chatbot_pipeline = None
_analysis_cache = {}


def get_sentence_model():
    """
    Load all-mpnet-base-v2 (higher quality, 768-dim embeddings).
    Falls back to all-MiniLM-L6-v2 if download fails.
    """
    """"""
    global _sentence_model
    if _sentence_model is None:
        for model_name in ('all-MiniLM-L6-v2'):
            try:
                _sentence_model = SentenceTransformer(model_name)
                print(f"[INFO] Loaded embedding model: {model_name}")
                break
            except Exception as e:
                print(f"[WARN] Could not load {model_name}: {e}")
    return _sentence_model

""" facebook/opt-350m"""
def get_chatbot():
    global _chatbot_pipeline
    if _chatbot_pipeline is None:
        try:
            _chatbot_pipeline = pipeline(
                "text-generation", model="LLaMA 3",
                max_new_tokens=200, do_sample=True,
                temperature=0.7, pad_token_id=50256
            )
        except Exception:
            _chatbot_pipeline = None
    return _chatbot_pipeline


# ══════════════════════════════════════════════════════
#  RULE 1 — ROLE SKILLS DATABASE
#  Strict dictionary: role → {core: [...], optional: [...]}
#  Core skills weight = 2  |  Optional weight = 1
# ══════════════════════════════════════════════════════

ROLE_SKILLS_DB = {
    "Data Scientist": {
        "core": [
            "Python", "SQL", "machine learning", "statistics", "pandas",
            "NumPy", "data analysis", "scikit-learn", "data visualization",
            "Jupyter", "hypothesis testing", "feature engineering"
        ],
        "optional": [
            "R", "TensorFlow", "PyTorch", "deep learning", "NLP",
            "A/B testing", "Spark", "Tableau", "Power BI", "matplotlib",
            "seaborn", "MLflow", "Airflow", "Hadoop", "Kafka", "XGBoost"
        ]
    },
    "Machine Learning Engineer": {
        "core": [
            "Python", "machine learning", "deep learning", "TensorFlow",
            "PyTorch", "MLOps", "Docker", "Git", "SQL", "scikit-learn",
            "model deployment", "REST API"
        ],
        "optional": [
            "C++", "Go", "Kubernetes", "Spark", "AWS", "Azure", "GCP",
            "Kafka", "feature engineering", "CUDA", "JAX", "MLflow",
            "Airflow", "Triton", "ONNX", "Redis", "Grafana"
        ]
    },
    "Data Engineer": {
        "core": [
            "Python", "SQL", "Apache Spark", "ETL", "data pipeline",
            "Airflow", "Kafka", "cloud computing", "data warehouse", "Git"
        ],
        "optional": [
            "Scala", "dbt", "Snowflake", "BigQuery", "Redshift",
            "Delta Lake", "Hadoop", "Flink", "Terraform", "Docker",
            "Kubernetes", "AWS", "Azure", "GCP", "PostgreSQL"
        ]
    },
    "Backend Engineer": {
        "core": [
            "Python", "REST API", "SQL", "Git", "Docker", "microservices",
            "Linux", "databases", "API design", "authentication"
        ],
        "optional": [
            "Java", "Go", "Node.js", "FastAPI", "Flask", "Django",
            "Spring Boot", "gRPC", "Kafka", "Redis", "PostgreSQL",
            "MongoDB", "Kubernetes", "AWS", "CI/CD", "GraphQL"
        ]
    },
    "Frontend Developer": {
        "core": [
            "HTML", "CSS", "JavaScript", "React", "TypeScript",
            "REST API", "Git", "responsive design", "UI components"
        ],
        "optional": [
            "Next.js", "Vue", "Angular", "Redux", "GraphQL",
            "Webpack", "Vite", "Jest", "accessibility", "SASS",
            "CSS-in-JS", "Figma", "performance optimization", "Node.js"
        ]
    },
    "Full Stack Developer": {
        "core": [
            "HTML", "CSS", "JavaScript", "React", "Node.js",
            "REST API", "SQL", "Git", "databases", "TypeScript"
        ],
        "optional": [
            "Next.js", "Vue", "Express", "MongoDB", "PostgreSQL",
            "Docker", "AWS", "GraphQL", "Redux", "Python",
            "Kubernetes", "CI/CD", "Tailwind CSS", "authentication"
        ]
    },
    "DevOps Engineer": {
        "core": [
            "Docker", "Kubernetes", "CI/CD", "Linux", "Terraform",
            "AWS", "Git", "scripting", "monitoring", "automation"
        ],
        "optional": [
            "Azure", "GCP", "Ansible", "Helm", "Prometheus", "Grafana",
            "Jenkins", "GitHub Actions", "Bash", "Python", "Go",
            "Nginx", "Vault", "ArgoCD", "Datadog", "ELK Stack"
        ]
    },
    "Site Reliability Engineer": {
        "core": [
            "Linux", "Kubernetes", "monitoring", "incident response",
            "Python", "Go", "SLO", "observability", "Docker", "CI/CD"
        ],
        "optional": [
            "Prometheus", "Grafana", "Terraform", "AWS", "chaos engineering",
            "Datadog", "OpenTelemetry", "Kafka", "Ansible", "Helm",
            "capacity planning", "Bash", "PagerDuty", "Elasticsearch"
        ]
    },
    "Cloud Architect": {
        "core": [
            "AWS", "cloud computing", "architecture", "security",
            "Terraform", "Docker", "Kubernetes", "networking", "IAM"
        ],
        "optional": [
            "Azure", "GCP", "serverless", "microservices", "cost optimization",
            "compliance", "CDN", "load balancing", "S3", "Lambda",
            "EC2", "VPC", "CloudFormation", "multi-cloud", "FinOps"
        ]
    },
    "Cybersecurity Analyst": {
        "core": [
            "network security", "penetration testing", "SIEM", "OWASP",
            "incident response", "vulnerability assessment", "cryptography",
            "Linux", "Python", "threat intelligence"
        ],
        "optional": [
            "Metasploit", "Burp Suite", "Wireshark", "MITRE ATT&CK",
            "SOC", "malware analysis", "forensics", "ISO 27001", "NIST",
            "firewall", "IDS/IPS", "compliance", "Splunk", "ethical hacking"
        ]
    },
    "iOS Developer": {
        "core": [
            "Swift", "Xcode", "UIKit", "SwiftUI", "iOS", "Objective-C",
            "Core Data", "REST API", "Git", "MVC"
        ],
        "optional": [
            "Combine", "async/await", "CloudKit", "TestFlight", "ARKit",
            "CoreML", "MVVM", "RxSwift", "CocoaPods", "SPM",
            "push notifications", "App Store Connect", "performance optimization"
        ]
    },
    "Android Developer": {
        "core": [
            "Kotlin", "Android", "Jetpack Compose", "Android Studio",
            "REST API", "Git", "MVVM", "Room", "Coroutines"
        ],
        "optional": [
            "Java", "Retrofit", "Hilt", "LiveData", "Navigation Component",
            "Firebase", "WorkManager", "ExoPlayer", "Gradle",
            "Play Store", "unit testing", "Espresso", "Material Design"
        ]
    },
    "Mobile Developer": {
        "core": [
            "React Native", "iOS", "Android", "JavaScript", "TypeScript",
            "REST API", "Git", "mobile UI", "state management"
        ],
        "optional": [
            "Flutter", "Dart", "Swift", "Kotlin", "Redux",
            "Firebase", "push notifications", "Expo", "performance optimization",
            "App Store", "Play Store", "animations"
        ]
    },
    "NLP Engineer": {
        "core": [
            "Python", "NLP", "transformers", "BERT", "machine learning",
            "PyTorch", "TensorFlow", "text preprocessing", "scikit-learn"
        ],
        "optional": [
            "Hugging Face", "GPT", "spaCy", "NLTK", "LangChain",
            "vector databases", "RAG", "fine-tuning", "tokenization",
            "sentiment analysis", "named entity recognition", "MLOps", "FastAPI"
        ]
    },
    "Computer Vision Engineer": {
        "core": [
            "Python", "computer vision", "OpenCV", "deep learning",
            "PyTorch", "TensorFlow", "image processing", "CNNs", "Git"
        ],
        "optional": [
            "YOLO", "object detection", "image segmentation", "GANs",
            "CUDA", "TensorRT", "ONNX", "data augmentation",
            "ResNet", "transfer learning", "NumPy", "scikit-learn"
        ]
    },
    "Data Analyst": {
        "core": [
            "SQL", "Excel", "Python", "data visualization", "statistics",
            "Tableau", "Power BI", "pandas", "business intelligence", "reporting"
        ],
        "optional": [
            "R", "Looker", "Metabase", "Google Analytics", "A/B testing",
            "NumPy", "matplotlib", "seaborn", "data cleaning",
            "dashboards", "data storytelling", "ETL", "BigQuery"
        ]
    },
    "Product Manager": {
        "core": [
            "product roadmap", "agile", "scrum", "user research",
            "data analysis", "stakeholder management", "JIRA", "OKR",
            "product strategy", "prioritization"
        ],
        "optional": [
            "Confluence", "Notion", "A/B testing", "SQL", "Figma",
            "user stories", "sprint planning", "KPI", "go-to-market",
            "competitor analysis", "customer discovery", "analytics"
        ]
    },
    "UX/UI Designer": {
        "core": [
            "Figma", "user research", "wireframing", "prototyping",
            "UX design", "UI design", "design systems", "usability testing",
            "accessibility", "WCAG"
        ],
        "optional": [
            "Adobe XD", "Sketch", "InVision", "motion design",
            "information architecture", "personas", "user flows",
            "visual design", "HTML", "CSS", "design thinking"
        ]
    },
    "Blockchain Engineer": {
        "core": [
            "Solidity", "Ethereum", "smart contracts", "Web3.js",
            "blockchain", "DeFi", "cryptography", "Git", "JavaScript"
        ],
        "optional": [
            "Rust", "Hardhat", "Truffle", "IPFS", "NFT",
            "MetaMask", "Solana", "Layer 2", "zero-knowledge proofs",
            "tokenomics", "Foundry", "OpenZeppelin", "React"
        ]
    },
    "Software Engineer": {
        "core": [
            "Python", "Java", "JavaScript", "Git", "data structures",
            "algorithms", "SQL", "REST API", "object-oriented programming",
            "testing", "code review"
        ],
        "optional": [
            "TypeScript", "C++", "Go", "Docker", "Kubernetes",
            "AWS", "Linux", "CI/CD", "microservices", "design patterns",
            "system design", "agile", "React", "Node.js"
        ]
    },
}

# Fallback when role is not in DB — derive skills generically using embeddings
_GENERIC_TECH_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "SQL",
    "Git", "Docker", "Kubernetes", "REST API", "machine learning", "deep learning",
    "cloud computing", "AWS", "Linux", "CI/CD", "microservices", "databases",
    "algorithms", "data structures", "system design", "agile", "testing",
]


# ─────────────────────────────────────────────
# Company Dataset (ML-matched, with LinkedIn links)
# ─────────────────────────────────────────────
COMPANY_DATASET = [
    {"company": "Google", "role": "Software Engineer",
     "domain": "search infrastructure reliability cloud AI algorithms Python Go C++",
     "description": "Design large-scale distributed systems, AI/ML services. Python, Go, C++, TensorFlow, Kubernetes, GCP. Strong algorithms required.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer&f_C=1441",
     "linkedin_company": "https://www.linkedin.com/company/google/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Google-Software-Engineer-Jobs-E9079.htm",
     "careers": "https://careers.google.com"},
    {"company": "Meta", "role": "Data Scientist",
     "domain": "social media data analytics machine learning NLP SQL Python Spark PyTorch",
     "description": "Analyze datasets, build recommendation models. Python, R, SQL, A/B testing, Spark, PyTorch. Design experiments through data.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Data+Scientist&f_C=10667",
     "linkedin_company": "https://www.linkedin.com/company/meta/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Meta-Data-Scientist-Jobs-E40772.htm",
     "careers": "https://metacareers.com"},
    {"company": "Amazon / AWS", "role": "Cloud Solutions Architect",
     "domain": "cloud AWS infrastructure DevOps serverless microservices EC2 Lambda Terraform",
     "description": "Architect enterprise cloud on AWS. EC2, EKS, Lambda, RDS, DynamoDB, CloudFormation. Lead cloud migration projects.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Cloud+Solutions+Architect&f_C=1586",
     "linkedin_company": "https://www.linkedin.com/company/amazon/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Amazon-Solutions-Architect-Jobs-E6036.htm",
     "careers": "https://amazon.jobs"},
    {"company": "Microsoft", "role": "Software Engineer",
     "domain": "cloud Azure .NET C# TypeScript React Office enterprise Python DevOps",
     "description": "Build enterprise software with C#, .NET, TypeScript, React, Azure. Office 365, Teams, GitHub.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer&f_C=1035",
     "linkedin_company": "https://www.linkedin.com/company/microsoft/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Microsoft-Software-Engineer-Jobs-E1651.htm",
     "careers": "https://careers.microsoft.com"},
    {"company": "OpenAI", "role": "ML Research Engineer",
     "domain": "artificial intelligence machine learning NLP LLM transformers deep learning RLHF PyTorch CUDA",
     "description": "Research and build foundation models. PyTorch, CUDA, JAX, distributed training, inference optimization.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=ML+Engineer&f_C=34702650",
     "linkedin_company": "https://www.linkedin.com/company/openai/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/OpenAI-Jobs-E3371290.htm",
     "careers": "https://openai.com/careers"},
    {"company": "Netflix", "role": "Senior Backend Engineer",
     "domain": "streaming backend microservices Java Python Kafka Cassandra Redis reliability",
     "description": "Design high-throughput microservices. Java, Python, Kafka, gRPC, Cassandra, Redis, AWS.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Backend+Engineer&f_C=165158",
     "linkedin_company": "https://www.linkedin.com/company/netflix/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Netflix-Software-Engineer-Jobs-E11891.htm",
     "careers": "https://jobs.netflix.com"},
    {"company": "Stripe", "role": "Full Stack Engineer",
     "domain": "fintech payments API Ruby Go TypeScript React PostgreSQL distributed",
     "description": "Build payment infrastructure. Ruby, Go, TypeScript, React, PostgreSQL, distributed systems.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer&f_C=671923",
     "linkedin_company": "https://www.linkedin.com/company/stripe/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Stripe-Software-Engineer-Jobs-E671923.htm",
     "careers": "https://stripe.com/jobs"},
    {"company": "Databricks", "role": "Data Platform Engineer",
     "domain": "Spark Delta Lake MLflow data engineering Python Scala SQL ETL cloud analytics",
     "description": "Build lakehouse platform with Spark, Delta Lake, MLflow. Distributed query, ML pipelines, cloud integrations.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Data+Engineer&f_C=10311690",
     "linkedin_company": "https://www.linkedin.com/company/databricks/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Databricks-Jobs-E1289087.htm",
     "careers": "https://databricks.com/company/careers"},
    {"company": "Airbnb", "role": "Frontend Engineer",
     "domain": "React TypeScript GraphQL CSS accessibility UI UX design system performance",
     "description": "Build world-class UIs. React, TypeScript, GraphQL, CSS-in-JS, accessibility, performance optimization.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Frontend+Engineer&f_C=391850",
     "linkedin_company": "https://www.linkedin.com/company/airbnb/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Airbnb-Software-Engineer-Jobs-E391850.htm",
     "careers": "https://careers.airbnb.com"},
    {"company": "CrowdStrike", "role": "Cybersecurity Analyst",
     "domain": "cybersecurity threat intelligence SIEM penetration testing OWASP NIST incident response malware",
     "description": "Hunt threats, investigate incidents, conduct vulnerability assessments, build detection rules.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Cybersecurity+Analyst&f_C=78083",
     "linkedin_company": "https://www.linkedin.com/company/crowdstrike/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/CrowdStrike-Jobs-E436241.htm",
     "careers": "https://crowdstrike.com/careers"},
    {"company": "Snowflake", "role": "Data Engineer",
     "domain": "data engineering SQL ETL dbt Spark Airflow Kafka Snowflake Python cloud analytics",
     "description": "Build scalable data pipelines. Snowflake, dbt, Airflow, Kafka, Python, star schemas.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Data+Engineer&f_C=3513273",
     "linkedin_company": "https://www.linkedin.com/company/snowflake-computing/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Snowflake-Data-Engineer-Jobs-E3513273.htm",
     "careers": "https://careers.snowflake.com"},
    {"company": "Spotify", "role": "Mobile Developer",
     "domain": "iOS Android Swift Kotlin React Native audio streaming Jetpack mobile",
     "description": "iOS (Swift) and Android (Kotlin) apps. Audio streaming, offline mode, Jetpack Compose.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Mobile+Developer&f_C=161357",
     "linkedin_company": "https://www.linkedin.com/company/spotify/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Spotify-Mobile-Developer-Jobs-E408251.htm",
     "careers": "https://lifeatspotify.com"},
    {"company": "Coinbase", "role": "Blockchain Engineer",
     "domain": "blockchain Solidity Ethereum DeFi smart contracts Web3 Rust cryptography",
     "description": "Build smart contracts in Solidity and Rust. DeFi protocols, NFT platforms, wallet integrations.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Blockchain+Engineer&f_C=9552792",
     "linkedin_company": "https://www.linkedin.com/company/coinbase/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Coinbase-Blockchain-Engineer-Jobs-E822923.htm",
     "careers": "https://coinbase.com/careers"},
    {"company": "LinkedIn", "role": "Site Reliability Engineer",
     "domain": "SRE reliability Kubernetes DevOps Go Python Terraform monitoring observability",
     "description": "Define SLOs, implement observability, manage on-call, lead incident response.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Site+Reliability+Engineer&f_C=1337",
     "linkedin_company": "https://www.linkedin.com/company/linkedin/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/LinkedIn-SRE-Jobs-E34865.htm",
     "careers": "https://careers.linkedin.com"},
    {"company": "Figma", "role": "UX/UI Designer",
     "domain": "Figma UX UI design systems user research prototyping wireframe accessibility WCAG",
     "description": "Create intuitive UX through research, wireframing, and Figma prototypes. WCAG accessibility.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=UX+Designer&f_C=3344592",
     "linkedin_company": "https://www.linkedin.com/company/figma/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Figma-Designer-Jobs-E3344592.htm",
     "careers": "https://figma.com/careers"},
    {"company": "Atlassian", "role": "Product Manager",
     "domain": "product management agile scrum OKR JIRA Confluence roadmap user research analytics SaaS",
     "description": "Define roadmaps, gather requirements. Agile, Scrum, OKR, JIRA, data analytics.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Product+Manager&f_C=3494",
     "linkedin_company": "https://www.linkedin.com/company/atlassian/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Atlassian-Product-Manager-Jobs-E1288259.htm",
     "careers": "https://www.atlassian.com/company/careers"},
    {"company": "Shopify", "role": "Full Stack Developer",
     "domain": "Ruby Rails React GraphQL MySQL e-commerce TypeScript JavaScript payments",
     "description": "Build commerce platform. Ruby on Rails, React, GraphQL, MySQL. Payment flows, merchant experiences.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Rails+Developer&f_C=246550",
     "linkedin_company": "https://www.linkedin.com/company/shopify/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Shopify-Software-Engineer-Jobs-E675523.htm",
     "careers": "https://shopify.com/careers"},
    {"company": "Uber", "role": "Backend Engineer",
     "domain": "Go Python Kafka Cassandra Redis real-time geospatial microservices distributed",
     "description": "Build real-time systems. Go, Python, Kafka, Cassandra, Redis. Sub-second latency, multi-region.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Backend+Engineer&f_C=2919",
     "linkedin_company": "https://www.linkedin.com/company/uber-com/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Uber-Software-Engineer-Jobs-E575263.htm",
     "careers": "https://www.uber.com/global/en/careers/"},
    {"company": "Palantir", "role": "Software Engineer",
     "domain": "Java TypeScript Spark data analytics enterprise defense intelligence platform",
     "description": "Build data integration platforms (Gotham, Foundry, AIP). Java, TypeScript, Spark.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=Software+Engineer&f_C=29172",
     "linkedin_company": "https://www.linkedin.com/company/palantir-technologies/jobs/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Palantir-Technologies-Software-Engineer-Jobs-E457.htm",
     "careers": "https://jobs.lever.co/palantir"},
    {"company": "Hugging Face", "role": "NLP Engineer",
     "domain": "NLP transformers BERT GPT PyTorch HuggingFace fine-tuning LLM RAG Python machine learning",
     "description": "Build NLP models, fine-tune transformers, deploy LLMs. PyTorch, HuggingFace, Python.",
     "linkedin_search": "https://www.linkedin.com/jobs/search/?keywords=NLP+Engineer",
     "linkedin_company": "https://www.linkedin.com/company/hugging-face/",
     "glassdoor": "https://www.glassdoor.com/Jobs/Hugging-Face-Jobs-E2347023.htm",
     "careers": "https://apply.workable.com/huggingface"},
]

# Course database for recommend_courses()
COURSE_DESCRIPTIONS = [
    {"title": "Machine Learning Specialization",           "platform": "Coursera",  "description": "supervised learning unsupervised learning reinforcement Python regression classification scikit-learn neural networks algorithms model training evaluation",          "url": "https://coursera.org/specializations/machine-learning-introduction"},
    {"title": "Deep Learning Specialization",              "platform": "Coursera",  "description": "neural networks convolutional recurrent transformers backpropagation TensorFlow Keras NLP sequences computer vision",                                                  "url": "https://coursera.org/specializations/deep-learning"},
    {"title": "IBM Data Science Certificate",              "platform": "Coursera",  "description": "Python pandas NumPy matplotlib SQL statistics hypothesis testing data wrangling data visualization Jupyter data analysis",                                          "url": "https://coursera.org/professional-certificates/ibm-data-science"},
    {"title": "Python Bootcamp Zero to Hero",              "platform": "Udemy",     "description": "Python programming scripting functions loops OOP file handling web scraping requests JSON REST API automation beginner advanced",                                   "url": "https://www.udemy.com/course/complete-python-bootcamp"},
    {"title": "The Web Developer Bootcamp",                "platform": "Udemy",     "description": "HTML CSS JavaScript Node.js Express MongoDB REST API React full stack web development authentication deployment",                                                    "url": "https://www.udemy.com/course/the-web-developer-bootcamp"},
    {"title": "React — The Complete Guide",                "platform": "Udemy",     "description": "React hooks state management Redux Toolkit React Router JSX components custom hooks testing TypeScript SPA frontend",                                              "url": "https://www.udemy.com/course/react-the-complete-guide-incl-redux"},
    {"title": "AWS Certified Solutions Architect",         "platform": "Udemy",     "description": "AWS EC2 S3 RDS DynamoDB Lambda VPC IAM CloudFormation EKS cloud architecture serverless high availability scalability",                                           "url": "https://www.udemy.com/course/aws-certified-solutions-architect-associate-saa-c03"},
    {"title": "Docker & Kubernetes Practical Guide",       "platform": "Udemy",     "description": "Docker containers images Kubernetes orchestration pods deployments services Helm CI CD DevOps containerization microservices",                                      "url": "https://www.udemy.com/course/docker-kubernetes-the-practical-guide"},
    {"title": "NLP Specialization",                        "platform": "Coursera",  "description": "NLP tokenization embeddings attention transformers BERT GPT sentiment text classification machine translation chatbot sequence models",                            "url": "https://coursera.org/specializations/natural-language-processing"},
    {"title": "SQL for Data Science",                      "platform": "Coursera",  "description": "SQL queries joins aggregation subqueries window functions indexes database design PostgreSQL MySQL data analysis",                                                  "url": "https://www.coursera.org/learn/sql-for-data-science"},
    {"title": "Cybersecurity Analyst Certificate",         "platform": "edX",       "description": "network security cryptography penetration testing OWASP ethical hacking firewalls SIEM incident response malware forensics vulnerability",                         "url": "https://www.edx.org/professional-certificate/ibm-cybersecurity-analyst"},
    {"title": "Data Engineering with Python",              "platform": "edX",       "description": "Apache Spark Kafka Airflow ETL data pipelines PySpark Delta Lake streaming data lake warehouse dbt",                                                               "url": "https://www.edx.org/professional-certificate/harvardx-data-engineering"},
    {"title": "iOS App Development with Swift",            "platform": "Coursera",  "description": "Swift Xcode UIKit SwiftUI Core Data networking iOS mobile app development MVC MVVM App Store",                                                                     "url": "https://developer.apple.com/tutorials"},
    {"title": "Android Development with Kotlin",           "platform": "Udemy",     "description": "Kotlin Android Jetpack Compose Room LiveData ViewModel Retrofit Coroutines Hilt Android Studio mobile app development",                                            "url": "https://www.udemy.com/course/android-kotlin-developer"},
    {"title": "Blockchain & Web3 Development",             "platform": "Coursera",  "description": "Solidity smart contracts Ethereum Hardhat MetaMask DeFi NFT Web3.js blockchain decentralized applications cryptography",                                           "url": "https://coursera.org/specializations/blockchain"},
    {"title": "DevOps Engineering on AWS",                 "platform": "Coursera",  "description": "CI CD Jenkins GitHub Actions Terraform Ansible Docker Kubernetes monitoring infrastructure as code GitOps deployment",                                             "url": "https://coursera.org/professional-certificates/aws-devops"},
    {"title": "Figma UI/UX Design Essentials",             "platform": "Udemy",     "description": "Figma wireframe prototype design system components user research accessibility UX UI design web mobile",                                                            "url": "https://www.udemy.com/course/figma-ux-ui-design-user-experience-tutorial-course"},
    {"title": "Statistics for Data Science",               "platform": "edX",       "description": "probability distributions hypothesis testing A/B testing regression statistical inference bayesian methods R Python",                                               "url": "https://www.edx.org/course/statistics-and-data-science"},
    {"title": "Agile Project Management with Scrum",       "platform": "Coursera",  "description": "agile scrum kanban sprint planning retrospective JIRA product roadmap OKR stakeholder management backlog prioritization",                                          "url": "https://www.coursera.org/learn/agile-development"},
    {"title": "TypeScript Complete Developer Guide",        "platform": "Udemy",     "description": "TypeScript types interfaces generics decorators advanced types JavaScript strongly typed Node.js React",                                                           "url": "https://www.udemy.com/course/typescript-the-complete-developers-guide"},
    {"title": "Data Structures and Algorithms Bootcamp",   "platform": "Udemy",     "description": "algorithms sorting searching trees graphs dynamic programming problem solving coding interview LeetCode",                                                          "url": "https://www.udemy.com/course/js-algorithms-and-data-structures-masterclass"},
    {"title": "Microsoft Azure Fundamentals",              "platform": "edX",       "description": "Azure cloud services VMs storage networking identity security compliance cost management Azure DevOps",                                                              "url": "https://www.edx.org/learn/azure/microsoft-azure-fundamentals-az-900-exam-prep"},
    {"title": "Apache Spark and Scala Certification",      "platform": "Udemy",     "description": "Spark Scala RDD DataFrame SQL streaming MLlib big data distributed computing Hadoop",                                                                               "url": "https://www.udemy.com/course/apache-spark-with-scala-hands-on-with-big-data"},
    {"title": "Google Cloud Professional Data Engineer",   "platform": "Coursera",  "description": "BigQuery Dataflow Pub/Sub Dataproc Cloud Storage GCP ETL machine learning cloud data engineering pipelines",                                                       "url": "https://coursera.org/professional-certificates/gcp-data-engineering"},
    {"title": "Penetration Testing and Ethical Hacking",   "platform": "Udemy",     "description": "penetration testing ethical hacking Metasploit Burp Suite network security Kali Linux OWASP web application security",                                             "url": "https://www.udemy.com/course/learn-ethical-hacking-from-scratch"},
]

YOUTUBE_TOPIC_SEEDS = [
    {"topic": "Python programming complete course tutorial",              "domain": "Python programming scripting automation"},
    {"topic": "Machine learning Python scikit-learn full course",         "domain": "machine learning algorithms model training scikit-learn"},
    {"topic": "Deep learning PyTorch TensorFlow neural networks course",  "domain": "deep learning neural networks PyTorch TensorFlow"},
    {"topic": "React JavaScript frontend web development course",         "domain": "React JavaScript frontend components hooks TypeScript"},
    {"topic": "Node.js Express REST API backend tutorial",                "domain": "Node.js Express backend REST API server JavaScript"},
    {"topic": "Docker Kubernetes DevOps containerization guide",          "domain": "Docker Kubernetes containers DevOps CI CD deployment"},
    {"topic": "AWS cloud computing services tutorial",                    "domain": "AWS cloud EC2 S3 Lambda serverless architecture"},
    {"topic": "SQL database queries tutorial for beginners",              "domain": "SQL database queries joins window functions"},
    {"topic": "Data structures algorithms interview preparation",         "domain": "algorithms data structures coding interview problem solving"},
    {"topic": "System design interview large scale distributed",          "domain": "system design distributed scalability architecture"},
    {"topic": "TypeScript complete developer guide",                      "domain": "TypeScript types interfaces generics strongly typed"},
    {"topic": "Data analysis pandas NumPy matplotlib Python",            "domain": "data analysis pandas NumPy matplotlib visualization"},
    {"topic": "NLP BERT transformers Hugging Face tutorial",             "domain": "NLP text BERT GPT transformers sentiment classification"},
    {"topic": "Cybersecurity ethical hacking penetration testing",       "domain": "cybersecurity hacking pentesting network OWASP"},
    {"topic": "iOS Swift app development complete course",                "domain": "iOS Swift Xcode mobile UIKit SwiftUI"},
    {"topic": "Android Kotlin Jetpack Compose tutorial",                 "domain": "Android Kotlin Jetpack Compose mobile development"},
    {"topic": "Terraform infrastructure as code AWS Azure",              "domain": "Terraform IaC cloud provisioning infrastructure automation"},
    {"topic": "Figma UI UX design complete course",                      "domain": "Figma UX UI design wireframe prototype user experience"},
    {"topic": "Apache Spark PySpark big data tutorial",                  "domain": "Spark PySpark big data ETL distributed processing"},
    {"topic": "Solidity Ethereum smart contracts blockchain course",      "domain": "Solidity Ethereum smart contracts blockchain Web3 DeFi"},
    {"topic": "Git GitHub version control workflow tutorial",             "domain": "Git GitHub version control branching workflow collaboration"},
    {"topic": "Statistics probability data science machine learning",    "domain": "statistics probability hypothesis testing A/B testing"},
    {"topic": "Agile Scrum project management tutorial",                 "domain": "agile scrum kanban project management sprint OKR"},
    {"topic": "Java Spring Boot microservices full course",              "domain": "Java Spring Boot microservices enterprise backend REST"},
    {"topic": "Kotlin Android app development complete guide",           "domain": "Kotlin Android app Jetpack Room Coroutines MVVM"},
]


# ══════════════════════════════════════════════════════
#  Text Utilities
# ══════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes):
    """Extract plain text from PDF bytes using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = "".join(page.get_text("text") + "\n" for page in doc)
    doc.close()
    return text.strip()


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[|●•·■◆→➤]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', text)
    return [s.strip() for s in parts if len(s.strip()) > 15]


def detect_sections(text):
    patterns = {
        'summary':        r'\b(summary|objective|profile|about me|overview)\b',
        'experience':     r'\b(experience|work history|employment|professional background)\b',
        'education':      r'\b(education|academic|university|degree|qualification)\b',
        'skills':         r'\b(skills|competencies|expertise|technical skills|proficiencies)\b',
        'projects':       r'\b(projects|portfolio|work samples|notable projects)\b',
        'certifications': r'\b(certifications|certificates|credentials|licenses)\b',
        'achievements':   r'\b(achievements|accomplishments|awards|honors)\b',
        'contact':        r'\b(contact|email|phone|linkedin|github|address)\b',
    }
    lower = text.lower()
    return {k: bool(re.search(p, lower)) for k, p in patterns.items()}


# ══════════════════════════════════════════════════════
#  RULE 1 — Role lookup: exact → partial → semantic
# ══════════════════════════════════════════════════════

def get_role_definition(job_role):
    """
    Return (matched_role_name, role_def_dict) for the given job role string.
    Priority: exact match → partial match → semantic embedding match → generic fallback.
    """
    jr_lower = job_role.lower().strip()

    # 1. Exact (case-insensitive)
    for key in ROLE_SKILLS_DB:
        if key.lower() == jr_lower:
            return key, ROLE_SKILLS_DB[key]

    # 2. Partial (role key is substring of user input or vice versa)
    for key in ROLE_SKILLS_DB:
        if key.lower() in jr_lower or jr_lower in key.lower():
            return key, ROLE_SKILLS_DB[key]

    # 3. Semantic — embed job_role vs all keys
    model = get_sentence_model()
    role_keys = list(ROLE_SKILLS_DB.keys())
    key_embs  = model.encode(role_keys, normalize_embeddings=True, batch_size=32)
    q_emb     = model.encode(job_role,  normalize_embeddings=True)
    sims      = cosine_similarity([q_emb], key_embs)[0]
    best_idx  = int(np.argmax(sims))
    best_sim  = float(sims[best_idx])

    if best_sim >= 0.45:                     # confident enough
        best_key = role_keys[best_idx]
        return best_key, ROLE_SKILLS_DB[best_key]

    # 4. Fallback — use generic tech skills
    generic = {
        "core":     _GENERIC_TECH_SKILLS[:10],
        "optional": _GENERIC_TECH_SKILLS[10:],
    }
    return job_role, generic


# ══════════════════════════════════════════════════════
#  RULE 2 — Hybrid skill matching: exact keyword + embeddings
# ══════════════════════════════════════════════════════

# Embedding similarity threshold for semantic match
_EMBED_THRESHOLD  = 0.52   # skill_emb vs resume_emb global cosine

def hybrid_skill_found(skill, resume_text_lower, resume_emb, model, skill_embs_cache):
    """
    Returns (found: bool, method: str, score: float).

    Exact keyword check  — fast O(n) substring search on lowercase text.
    Embedding check      — cosine similarity between skill embedding and
                           overall resume embedding (pre-computed).
    A skill is found if EITHER test passes.
    """
    # ── Exact / keyword match ──────────────────────────────────────────
    skill_l = skill.lower()
    # Try bare substring first (handles multi-word skills like "machine learning")
    if skill_l in resume_text_lower:
        return True, 'exact', 1.0
    # Also check common abbreviations (Python→py, JavaScript→js, etc.) – basic
    abbrev_map = {
        'javascript': 'js', 'typescript': 'ts', 'python': 'py',
        'machine learning': 'ml', 'natural language processing': 'nlp',
        'artificial intelligence': 'ai', 'continuous integration': 'ci',
        'continuous delivery': 'cd', 'ci/cd': 'cicd',
        'application programming interface': 'api',
    }
    if skill_l in abbrev_map and abbrev_map[skill_l] in resume_text_lower:
        return True, 'abbrev', 0.95

    # ── Embedding / semantic match ─────────────────────────────────────
    if skill not in skill_embs_cache:
        skill_embs_cache[skill] = model.encode(skill, normalize_embeddings=True)
    skill_emb = skill_embs_cache[skill]
    sim = float(cosine_similarity([skill_emb], [resume_emb])[0][0])
    if sim >= _EMBED_THRESHOLD:
        return True, 'semantic', sim

    return False, 'none', sim


# ══════════════════════════════════════════════════════
#  RULE 3 + RULE 5 — extract_resume_skills
#  ONLY evaluates skills that belong to the selected role.
#  Returns categorized dicts for matched and missing.
# ══════════════════════════════════════════════════════

def extract_resume_skills(resume_text, role_def):
    """
    Strict role-filtered skill extraction using hybrid matching.

    Returns:
        matched  = {'core': [...], 'optional': [...]}
        missing  = {'core': [...], 'optional': [...]}
        details  = {skill: {found, method, score}, ...}
    """
    model            = get_sentence_model()
    resume_text_l    = resume_text.lower()
    resume_emb       = model.encode(resume_text[:3000], normalize_embeddings=True)
    skill_embs_cache = {}

    matched = {'core': [], 'optional': []}
    missing = {'core': [], 'optional': []}
    details = {}

    for tier in ('core', 'optional'):
        for skill in role_def.get(tier, []):
            found, method, score = hybrid_skill_found(
                skill, resume_text_l, resume_emb, model, skill_embs_cache
            )
            details[skill] = {'found': found, 'method': method, 'score': round(score, 3), 'tier': tier}
            if found:
                matched[tier].append(skill)
            else:
                missing[tier].append(skill)

    return matched, missing, details


# ══════════════════════════════════════════════════════
#  RULE 4 — Weighted scoring
#  core skill weight = 2  |  optional weight = 1
# ══════════════════════════════════════════════════════

WEIGHT_CORE     = 2
WEIGHT_OPTIONAL = 1

def compute_weighted_match(matched, role_def):
    """
    Returns weighted match percentage (0-100).
    core_matched × 2 + optional_matched × 1
    ─────────────────────────────────────────
    total_core   × 2 + total_optional   × 1
    """
    core_total     = len(role_def.get('core', []))
    optional_total = len(role_def.get('optional', []))
    denom = core_total * WEIGHT_CORE + optional_total * WEIGHT_OPTIONAL
    if denom == 0:
        return 0.0

    core_hit     = len(matched.get('core', []))
    optional_hit = len(matched.get('optional', []))
    numer = core_hit * WEIGHT_CORE + optional_hit * WEIGHT_OPTIONAL

    return round((numer / denom) * 100, 1)


def compute_resume_score(resume_text, matched, missing, sections, weighted_match_pct):
    """
    Resume quality score 0–100.
    50 pts  → weighted role skill match (core/optional)
    25 pts  → section completeness
    15 pts  → quantified achievements + action verbs
    10 pts  → word count / content depth
    """
    score = 0.0

    # ── 50 pts: weighted skill match ──
    score += weighted_match_pct * 0.50

    # ── 25 pts: section completeness ──
    must_have = ['experience', 'education', 'skills', 'contact']
    nice_have = ['summary', 'projects', 'certifications', 'achievements']
    score += min(25, sum(5 for s in must_have if sections.get(s)) +
                     sum(1.25 for s in nice_have if sections.get(s)))

    # ── 10 pts: quantified achievements ──
    quant = len(re.findall(r'\b\d+%|\$\d+|\d+x|\d+ years|\d+\+', resume_text))
    score += min(10, quant * 1.8)

    # ── 5 pts: action verbs ──
    verbs = ['led','built','designed','improved','increased','reduced','launched',
             'managed','developed','implemented','architected','optimized','deployed']
    score += min(5, sum(1 for v in verbs if v in resume_text.lower()))

    # ── 10 pts: word count ──
    wc = len(resume_text.split())
    score += 10 if 350 <= wc <= 900 else (6 if wc > 900 else (3 if wc >= 150 else 0))

    return round(min(100, max(0, score)))


def compute_ats_score(resume_text, sections, matched):
    """ATS compatibility score 0-100."""
    score = 0.0
    has_email   = bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', resume_text))
    has_phone   = bool(re.search(r'[\+\(]?[0-9][0-9\s\-\(\)]{7,}[0-9]', resume_text))
    has_linkedin = bool(re.search(r'linkedin\.com|LinkedIn', resume_text, re.IGNORECASE))
    score += has_email * 10 + has_phone * 8 + has_linkedin * 7

    score += min(24, sum(8 for s in ['experience', 'education', 'skills'] if sections.get(s)))

    all_matched = matched.get('core', []) + matched.get('optional', [])
    score += min(20, len(all_matched) * 1.2)

    bullets = len(re.findall(r'^[\s]*[-•*▪►]', resume_text, re.MULTILINE))
    score += 8 if bullets >= 5 else (4 if bullets >= 2 else 0)
    dates = len(re.findall(
        r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b|\b\d{4}\s*[-–]\s*(\d{4}|present)\b',
        resume_text, re.IGNORECASE
    ))
    score += min(8, dates * 2)
    wc = len(resume_text.split())
    score += 8 if wc >= 250 else (4 if wc >= 100 else 0)
    return round(min(100, max(0, score)))


# ─────────────────────────────────────────────
# Company Recommendations (ML cosine similarity)
# ─────────────────────────────────────────────

def recommend_companies(resume_text, job_role, matched, top_n=5):
    model = get_sentence_model()
    all_matched = matched.get('core', []) + matched.get('optional', [])
    query = f"{job_role} {' '.join(all_matched)} {resume_text[:800]}"
    q_emb = model.encode(query, normalize_embeddings=True)
    co_texts = [f"{c['domain']} {c['description']}" for c in COMPANY_DATASET]
    co_embs  = model.encode(co_texts, normalize_embeddings=True, batch_size=32)
    sims     = cosine_similarity([q_emb], co_embs)[0]
    top_idx  = np.argsort(sims)[::-1][:top_n]
    return [{
        "company":          COMPANY_DATASET[i]['company'],
        "role":             COMPANY_DATASET[i]['role'],
        "match_score":      round(float(sims[i]) * 100, 1),
        "linkedin_search":  COMPANY_DATASET[i]['linkedin_search'],
        "linkedin_company": COMPANY_DATASET[i]['linkedin_company'],
        "glassdoor":        COMPANY_DATASET[i]['glassdoor'],
        "careers":          COMPANY_DATASET[i]['careers'],
        "snippet":          COMPANY_DATASET[i]['description'][:160] + "..."
    } for i in top_idx]


# ─────────────────────────────────────────────
# YouTube Recommendations (gap-driven, semantic)
# ─────────────────────────────────────────────

def recommend_youtube(missing, top_n=5):
    """Generate YouTube recommendations targeting missing skills."""
    all_missing = missing.get('core', []) + missing.get('optional', [])
    if not all_missing:
        return []
    model     = get_sentence_model()
    gap_query = "learn " + " ".join(all_missing[:12])
    q_emb     = model.encode(gap_query, normalize_embeddings=True)
    d_embs    = model.encode([s['domain'] for s in YOUTUBE_TOPIC_SEEDS], normalize_embeddings=True, batch_size=32)
    sims      = cosine_similarity([q_emb], d_embs)[0]
    top_idx   = np.argsort(sims)[::-1][:top_n]
    return [{
        "title":       YOUTUBE_TOPIC_SEEDS[i]['topic'].replace("tutorial","").replace("course","").replace("complete","").strip().title(),
        "youtube_url": "https://www.youtube.com/results?search_query=" + urllib.parse.quote(YOUTUBE_TOPIC_SEEDS[i]['topic']),
        "relevance":   round(float(sims[i]) * 100, 1),
    } for i in top_idx]


# ─────────────────────────────────────────────
# Course Recommendations (per missing skill)
# ─────────────────────────────────────────────

def recommend_courses(missing):
    """
    For each missing skill (core first, then optional),
    find top 2 courses + 1 YouTube link using NLP similarity.
    Returns { skill_name: [ {title, platform, url, relevance} ] }
    """
    all_missing = missing.get('core', []) + missing.get('optional', [])
    if not all_missing:
        return {}

    model       = get_sentence_model()
    course_txts = [f"{c['title']} {c['description']}" for c in COURSE_DESCRIPTIONS]
    course_embs = model.encode(course_txts, normalize_embeddings=True, batch_size=32)

    result = {}
    for skill in all_missing[:12]:
        s_emb = model.encode(f"{skill} course tutorial learn", normalize_embeddings=True)
        sims  = cosine_similarity([s_emb], course_embs)[0]
        top2  = np.argsort(sims)[::-1][:2]
        courses = []
        for idx in top2:
            c = COURSE_DESCRIPTIONS[idx]
            courses.append({"title": c['title'], "platform": c['platform'],
                            "url": c['url'], "relevance": round(float(sims[idx]) * 100, 1)})
        # Dynamic YouTube link
        yt_q = urllib.parse.quote(f"{skill} tutorial")
        courses.append({"title": f"{skill} — Free Tutorial",
                        "platform": "YouTube",
                        "url": f"https://www.youtube.com/results?search_query={yt_q}",
                        "relevance": 78})
        result[skill] = courses
    return result


# ─────────────────────────────────────────────
# Improvement Suggestions
# ─────────────────────────────────────────────

def generate_suggestions(resume_text, sections, missing, weighted_match_pct, ats_score):
    sug = []
    core_missing = missing.get('core', [])
    opt_missing  = missing.get('optional', [])

    if core_missing:
        sug.append({"priority": "High", "category": "Critical Skill Gaps",
            "text": f"You are missing {len(core_missing)} CORE skill(s) for this role: "
                    f"{', '.join(core_missing[:5])}{'...' if len(core_missing)>5 else ''}. "
                    "Add projects or certifications that demonstrate these."})
    if opt_missing:
        sug.append({"priority": "Medium", "category": "Optional Skill Gaps",
            "text": f"{len(opt_missing)} optional skill(s) missing: "
                    f"{', '.join(opt_missing[:5])}{'...' if len(opt_missing)>5 else ''}. "
                    "Adding even 2–3 of these will significantly boost your profile."})

    if weighted_match_pct < 50:
        sug.append({"priority": "High", "category": "Low Role Match",
            "text": f"Role match is only {weighted_match_pct}%. Focus on building the core skills for this specific role before applying."})

    if not sections.get('summary'):
        sug.append({"priority": "High", "category": "Missing Summary",
            "text": "Add a Professional Summary (3–4 lines) mentioning your target role and key strengths."})
    if not sections.get('achievements'):
        sug.append({"priority": "High", "category": "Quantified Achievements",
            "text": "Add an Achievements section with 3–5 measurable results (%, $, team size, timelines)."})

    quant = len(re.findall(r'\b\d+%|\$\d+|\d+x|\d+ years|\d+\+', resume_text))
    if quant < 3:
        sug.append({"priority": "High", "category": "Weak Impact Language",
            "text": f"Only {quant} quantified result(s) found. Add numbers to at least 5 bullet points."})

    if not re.search(r'linkedin\.com|LinkedIn', resume_text, re.IGNORECASE):
        sug.append({"priority": "High", "category": "Online Presence",
            "text": "Add your LinkedIn profile URL. 87% of recruiters check it before interviewing."})

    if ats_score < 60:
        sug.append({"priority": "High", "category": "ATS Compatibility",
            "text": "ATS score is below average. Use standard section headings, avoid tables/graphics, and ensure skills match exact keywords from job descriptions."})

    wc = len(resume_text.split())
    if wc < 250:
        sug.append({"priority": "Medium", "category": "Content Depth",
            "text": f"Resume is too brief ({wc} words). Expand each role to 3–5 bullets. Aim for 400–700 words."})

    return sorted(sug, key=lambda x: {"High": 0, "Medium": 1, "Low": 2}.get(x['priority'], 3))


# ─────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────

def chat_response(user_message, cache_data):
    chatbot  = get_chatbot()
    job_role = cache_data.get('job_role', 'your target role')
    matched  = cache_data.get('matched_skills', {})
    missing  = cache_data.get('missing_skills', {})
    rs       = cache_data.get('resume_score', 'N/A')
    ats      = cache_data.get('ats_score', 'N/A')
    wm       = cache_data.get('weighted_match_pct', 'N/A')

    matched_list = matched.get('core', []) + matched.get('optional', [])
    missing_list = missing.get('core', []) + missing.get('optional', [])

    prompt = (f"Career advisor. Role: {job_role}. Resume: {rs}/100. ATS: {ats}/100. "
              f"Role match: {wm}%. Skills found: {', '.join(matched_list[:8])}. "
              f"Missing: {', '.join(missing_list[:6])}.\nUser: {user_message}\nAdvisor:")

    if chatbot:
        try:
            result    = chatbot(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)
            generated = result[0]['generated_text']
            if "Advisor:" in generated:
                resp = generated.split("Advisor:")[-1].strip()
                return ' '.join(re.split(r'(?<=[.!?])\s+', resp)[:4]).strip()
        except Exception:
            pass

    # Smart fallback using embeddings
    model  = get_sentence_model()
    q_emb  = model.encode(user_message, normalize_embeddings=True)
    routes = {
        "improve":   "how to improve resume",
        "missing":   "what skills am I missing",
        "companies": "which companies should I apply to",
        "score":     "explain resume and ATS scores",
        "courses":   "what courses should I take",
        "interview": "how to prepare for interview",
        "cover":     "how to write cover letter",
    }
    r_embs = model.encode(list(routes.values()), normalize_embeddings=True)
    best   = list(routes.keys())[int(np.argmax(cosine_similarity([q_emb], r_embs)[0]))]
    top_co = ', '.join([f"{c['role']} at {c['company']}"
                        for c in cache_data.get('companies', [])[:3]])

    return {
        "improve":   f"Resume score is {rs}/100 with {wm}% role match for '{job_role}'. Priority: address the {len(missing.get('core',[]))} CORE missing skills first, then add quantifiable achievements.",
        "missing":   f"Core skills missing for '{job_role}': {', '.join(missing.get('core',[])[:6]) or 'None!'}. Optional: {', '.join(missing.get('optional',[])[:4]) or 'None!'}. See the Courses section to close these gaps.",
        "companies": f"Top companies for '{job_role}': {top_co or 'run full analysis first'}. All have LinkedIn apply links in the Companies tab.",
        "score":     f"Resume Score {rs}/100 = 50% role skill match + 25% sections + 15% achievements + 10% length. Role match {wm}% = weighted (core×2, optional×1) skill coverage.",
        "courses":   f"Courses are listed per missing skill in the Courses section. Core gaps ({', '.join(missing.get('core',[])[:3])}) are the highest priority.",
        "interview": f"For '{job_role}': use STAR method. Prepare examples for each matched skill. Research the company's tech stack. Practice system design if applicable.",
        "cover":     f"Use the Cover Letter section. Select a company, fill your details, and get a role-specific cover letter.",
    }.get(best, f"Based on your '{job_role}' resume (Score: {rs}/100, Role Match: {wm}%), ask me about missing skills, courses, companies, or interview prep.")


# ─────────────────────────────────────────────
# Cover Letter Generator
# ─────────────────────────────────────────────

def generate_cover_letter(company, role, name, email, years, matched, achievement, resume_text):
    company_entry = next((c for c in COMPANY_DATASET if c['company'].lower() in company.lower()), None)
    company_desc  = company_entry['description'][:180] if company_entry else "an industry-leading organization"
    all_matched   = matched.get('core', []) + matched.get('optional', [])
    skills_str    = ", ".join(all_matched[:5]) if all_matched else "software engineering"
    achievement_text = achievement or "delivering measurable impact"

    best_sentence = ""
    if resume_text:
        model     = get_sentence_model()
        sentences = split_sentences(resume_text)[:20]
        if sentences:
            role_emb  = model.encode(role + " " + company, normalize_embeddings=True)
            sent_embs = model.encode(sentences, normalize_embeddings=True, batch_size=32)
            best_sentence = sentences[int(np.argmax(cosine_similarity([role_emb], sent_embs)[0]))]
            best_sentence = re.sub(r'\s+', ' ', best_sentence[:200]).strip()

    return f"""[Your Address]
[City, State, ZIP]
[Date]

Hiring Manager
{company}
[Company Address]

Dear Hiring Manager,

I am writing to express my strong interest in the {role} position at {company}. With {years} years of experience in {skills_str}, I am confident I can make a meaningful contribution to your team.

{company} stands out to me because: {company_desc.rstrip('.')}. My background aligns directly with these goals.

Most notably, {achievement_text}. {best_sentence}

My core strengths include {skills_str}. I am drawn to this role because it combines the technical challenges I excel at with the collaborative culture I thrive in.

I would welcome the opportunity to discuss how I can contribute to {company}'s continued growth.

Sincerely,
{name}
{email}
[LinkedIn Profile URL]
[Portfolio / GitHub URL]""".strip()


# ─────────────────────────────────────────────
# PDF Report Generator
# ─────────────────────────────────────────────

def generate_pdf_report(data):
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    S    = getSampleStyleSheet()
    T    = ParagraphStyle('T',  parent=S['Heading1'], fontSize=20, spaceAfter=8,  textColor=colors.HexColor('#1a1a2e'))
    H2   = ParagraphStyle('H2', parent=S['Heading2'], fontSize=13, spaceAfter=5,  textColor=colors.HexColor('#0f3460'), spaceBefore=10)
    B    = ParagraphStyle('B',  parent=S['Normal'],   fontSize=10, spaceAfter=4)
    HL   = ParagraphStyle('HL', parent=S['Normal'],   fontSize=11, spaceAfter=4,  textColor=colors.HexColor('#0f3460'), fontName='Helvetica-Bold')
    RED  = ParagraphStyle('RD', parent=S['Normal'],   fontSize=10, spaceAfter=4,  textColor=colors.HexColor('#c0392b'))

    matched = data.get('matched_skills', {})
    missing = data.get('missing_skills', {})

    story = [
        Paragraph("AI Resume Intelligence Report", T),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor('#0f3460')),
        Spacer(1, 0.1 * inch),
        Paragraph(f"Target Role: <b>{data.get('job_role','N/A')}</b>   (matched to DB: <b>{data.get('matched_role','N/A')}</b>)", HL),
        Paragraph(f"Weighted Role Match: <b>{data.get('weighted_match_pct','N/A')}%</b>   |   Resume Score: <b>{data.get('resume_score','N/A')}/100</b>   |   ATS Score: <b>{data.get('ats_score','N/A')}/100</b>", HL),
        Spacer(1, 0.1 * inch),
    ]

    # Matched skills
    core_m = matched.get('core', [])
    opt_m  = matched.get('optional', [])
    if core_m or opt_m:
        story.append(Paragraph("Skills You Have for This Role", H2))
        if core_m:
            story.append(Paragraph(f"Core: {', '.join(core_m)}", B))
        if opt_m:
            story.append(Paragraph(f"Optional: {', '.join(opt_m)}", B))
        story.append(Spacer(1, 0.08 * inch))

    # Missing skills
    core_ms = missing.get('core', [])
    opt_ms  = missing.get('optional', [])
    if core_ms or opt_ms:
        story.append(Paragraph("Missing Skills (for target role)", H2))
        for s in core_ms:
            story.append(Paragraph(f"✗ [CORE] {s}", RED))
        for s in opt_ms[:8]:
            story.append(Paragraph(f"✗ [Optional] {s}", B))
        story.append(Spacer(1, 0.08 * inch))

    # Courses
    if data.get('course_recommendations'):
        story.append(Paragraph("Course Recommendations (per missing skill)", H2))
        for skill, courses in list(data['course_recommendations'].items())[:8]:
            story.append(Paragraph(f"<b>{skill}:</b>", B))
            for c in courses[:2]:
                story.append(Paragraph(f"  → {c['title']} [{c['platform']}] — {c['url']}", B))
        story.append(Spacer(1, 0.08 * inch))

    # Companies
    if data.get('companies'):
        story.append(Paragraph("Top Company Matches — Apply on LinkedIn", H2))
        for c in data['companies']:
            story.append(Paragraph(f"• <b>{c['role']}</b> @ {c['company']} — {c['match_score']}% | {c['linkedin_search']}", B))
        story.append(Spacer(1, 0.08 * inch))

    # Suggestions
    if data.get('suggestions'):
        story.append(Paragraph("Improvement Suggestions", H2))
        for s in data['suggestions']:
            col = '#c0392b' if s['priority']=='High' else ('#e67e22' if s['priority']=='Medium' else '#27ae60')
            story.append(Paragraph(
                f"[{s['priority']}] <b>{s['category']}</b>: {s['text']}",
                ParagraphStyle('P', parent=S['Normal'], fontSize=10, textColor=colors.HexColor(col))
            ))

    story += [
        Spacer(1, 0.2 * inch),
        HRFlowable(width="100%", thickness=1, color=colors.grey),
        Paragraph("Generated by AI Resume Intelligence System",
                  ParagraphStyle('F', parent=S['Normal'], fontSize=8, textColor=colors.grey)),
    ]
    doc.build(story)
    buf.seek(0)
    return buf


# ══════════════════════════════════════════════════════
#  Flask Routes
# ══════════════════════════════════════════════════════

@app.route('/')
def index():
    roles = sorted(ROLE_SKILLS_DB.keys())
    return render_template('index.html', roles=roles)


@app.route('/analyze', methods=['POST'])
def analyze():
    job_role = request.form.get('job_role', '').strip()
    if not job_role:
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error="Please enter your target job role.")

    if 'resume' not in request.files or not request.files['resume'].filename:
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error="Please select a PDF file to upload.")

    file = request.files['resume']
    if not file.filename.lower().endswith('.pdf'):
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error="Only PDF files are supported.")

    try:
        raw_text = extract_text_from_pdf(file.read())
    except Exception as e:
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error=f"Could not read PDF: {e}")

    if len(raw_text.strip()) < 50:
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error="Could not extract text. Ensure the PDF has selectable text.")

    resume_text = clean_text(raw_text)

    # ── Core analysis pipeline ──────────────────────────
    matched_role, role_def   = get_role_definition(job_role)
    matched, missing, details = extract_resume_skills(resume_text, role_def)
    weighted_match_pct       = compute_weighted_match(matched, role_def)
    sections                 = detect_sections(resume_text)
    resume_score             = compute_resume_score(resume_text, matched, missing, sections, weighted_match_pct)
    ats_score                = compute_ats_score(resume_text, sections, matched)
    companies                = recommend_companies(resume_text, job_role, matched, top_n=5)
    youtube_courses          = recommend_youtube(missing, top_n=5)
    course_recs              = recommend_courses(missing)
    suggestions              = generate_suggestions(resume_text, sections, missing, weighted_match_pct, ats_score)

    uid = str(abs(hash(resume_text[:200] + job_role)))
    _analysis_cache[uid] = {
        "text":                  resume_text,
        "filename":              secure_filename(file.filename),
        "job_role":              job_role,
        "matched_role":          matched_role,
        "role_def":              role_def,
        "matched_skills":        matched,
        "missing_skills":        missing,
        "skill_details":         details,
        "weighted_match_pct":    weighted_match_pct,
        "resume_score":          resume_score,
        "ats_score":             ats_score,
        "sections":              sections,
        "companies":             companies,
        "youtube_courses":       youtube_courses,
        "course_recommendations": course_recs,
        "suggestions":           suggestions,
    }
    return redirect(url_for('result', uid=uid))


@app.route('/result/<uid>')
def result(uid):
    data = _analysis_cache.get(uid)
    if not data:
        return render_template('index.html', roles=sorted(ROLE_SKILLS_DB.keys()),
                               error="Session expired. Please re-upload your resume.")
    return render_template('result.html', uid=uid, data=data)


# ── JSON endpoints ──

@app.route('/chat', methods=['POST'])
def chat():
    body = request.json or {}
    msg  = body.get('message', '').strip()
    uid  = body.get('upload_id', '')
    if not msg:
        return jsonify({'error': 'Empty message'}), 400
    return jsonify({'success': True, 'response': chat_response(msg, _analysis_cache.get(uid, {}))})


@app.route('/download')
def download():
    uid = request.args.get('uid', '')
    fmt = request.args.get('format', 'pdf')
    data = _analysis_cache.get(uid)
    if not data or 'resume_score' not in data:
        return jsonify({'error': 'No analysis found.'}), 404

    if fmt == 'txt':
        matched = data.get('matched_skills', {})
        missing = data.get('missing_skills', {})
        lines = [
            "=" * 60, "AI RESUME INTELLIGENCE REPORT", "=" * 60,
            f"Target Role      : {data.get('job_role','N/A')}",
            f"Matched DB Role  : {data.get('matched_role','N/A')}",
            f"Weighted Match   : {data.get('weighted_match_pct','N/A')}%",
            f"Resume Score     : {data.get('resume_score','N/A')}/100",
            f"ATS Score        : {data.get('ats_score','N/A')}/100",
            "", "CORE SKILLS MATCHED:",
            ", ".join(matched.get('core', [])) or "None",
            "", "OPTIONAL SKILLS MATCHED:",
            ", ".join(matched.get('optional', [])) or "None",
            "", "CORE SKILLS MISSING (CRITICAL):",
        ]
        for s in missing.get('core', []):
            lines.append(f"  ✗ {s}")
        lines += ["", "OPTIONAL SKILLS MISSING:"]
        for s in missing.get('optional', []):
            lines.append(f"  ✗ {s}")
        lines += ["", "COURSE RECOMMENDATIONS:"]
        for skill, courses in list(data.get('course_recommendations', {}).items())[:8]:
            lines.append(f"  {skill}:")
            for c in courses[:2]:
                lines.append(f"    → {c['title']} [{c['platform']}] {c['url']}")
        lines += ["", "TOP COMPANY MATCHES:"]
        for c in data.get('companies', []):
            lines.append(f"  • {c['role']} @ {c['company']} ({c['match_score']}%) → {c['linkedin_search']}")
        buf = io.BytesIO("\n".join(lines).encode('utf-8'))
        buf.seek(0)
        return send_file(buf, as_attachment=True, download_name='resume_report.txt', mimetype='text/plain')
    else:
        return send_file(generate_pdf_report(data), as_attachment=True,
                         download_name='resume_report.pdf', mimetype='application/pdf')


@app.route('/cover-letter', methods=['POST'])
def cover_letter():
    body    = request.json or {}
    uid     = body.get('uid', '')
    company = body.get('company', '').strip()
    role    = body.get('role', '').strip()
    name    = body.get('name', 'Your Name').strip()
    email   = body.get('email', 'your@email.com').strip()
    years   = body.get('years_exp', '3+').strip()
    achieve = body.get('achievement', '').strip()
    if not company or not role:
        return jsonify({'error': 'Company and role are required.'}), 400
    cache = _analysis_cache.get(uid, {})
    letter = generate_cover_letter(company, role, name, email, years,
                                   cache.get('matched_skills', {}), achieve,
                                   cache.get('text', ''))
    return jsonify({'success': True, 'cover_letter': letter})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
