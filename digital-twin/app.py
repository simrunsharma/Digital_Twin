"""
Deploying Digital Twin: Built from Digital Twin IPYNB
"""
import os
from openai import OpenAI
# from IPython.display import display, Markdown
import gradio as gr 
# from dotenv import load_dotenv
# import json
import uuid
# import random
# import requests
import chromadb
from pprint import pprint


#Setup-------------

#no loading the environment needed because its app.py
#You don't need it in production, hugging face already does it, but ipynb its to load into local machine
# load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#debugging if you can't find the openai key:
if OPENAI_API_KEY is None:
    raise Exception ( "API Key is missing")
else:
    print(OPENAI_API_KEY[:8])

client = OpenAI() 

#Document Overview------------------
document_experience = """


Name: Simrun Sharma

Current Role:
Associate Research Analyst / Data Scientist at CNA

Professional Experience

Center for Naval Analyses (CNA)
Associate Research Analyst / Data Scientist
May 2024 – Present
Washington, DC

Projects:
- Developed an NLP pipeline using regex, text processing, and information extraction to categorize research projects for budgeting and funding analysis.
- Performed strategic analysis of historical funding allocations to identify research areas receiving the greatest investment.
- Designed and implemented a Sparing Tool using a greedy optimization algorithm to determine optimal inventory levels for U.S. Air Force bases.
- Modeled backorders using a continuous Poisson distribution and iteratively added spares to satisfy target Non-Mission Capable (NMC) rates while remaining within budget constraints.
- Led development of a synthetic survey generation framework using web scraping, embedding models, CrewAI agents, vector databases, Retrieval-Augmented Generation (RAG), and Large Language Models (LLMs).
- Built an end-to-end AI pipeline capable of generating realistic synthetic survey responses.

Partners In Health
Co-Lead Advocacy Chair
September 2022 – February 2023

- Advocated for global health legislation through congressional meetings, callathons, and legislative outreach.
- Secured 47 co-signers for the NOVID Act.
- Worked on the Paul Farmer Memorial Resolution supporting increased global health funding.

University of Florida Center for Undergraduate Research
Director of Internal Affairs
March 2021 – February 2023

- Represented approximately 35,000 undergraduate students.
- Connected students with faculty research opportunities.
- Organized research workshops and student research showcases.

Equal Access Clinic Network
Social Work Clinical Coordinator
2021 – 2023

- Cleaned and standardized patient data profiles.
- Improved spreadsheet scheduling workflows for clinic staff.
- Created a bilingual community resource guide by consolidating over ten years of local organizational data.

University of Florida College of Public Health and Health Professions
Assistant Director
May 2022 – December 2022

- Coordinated communication among more than 25 research labs, colleges, and clinics.
- Organized a multidisciplinary health professions fair and student panels.

University of Florida Physicians Internal Medicine
Treasurer, MC, and Event Planner
November 2022

- Planned and hosted a physician networking event celebrating Diwali involving physicians from UF Health, the VA, and Northwest Hospital.

Planned Parenthood Federation of America
Student Volunteer
2021 – September 2022

- Participated in legislative advocacy, reproductive justice initiatives, petitions, and community organizing.

University of Florida
General Chemistry Teaching Assistant
January 2019 – January 2022

- Led weekly discussion sections.
- Created review sessions and supported undergraduate student learning.
"""

document_education = """
Name: Simrun Sharma

Education

Duke University
Master of Interdisciplinary Data Science (MIDS)
August 2023 – May 2025

Degree Focus:
- Data Science
- Machine Learning

Relevant Coursework:
- Modeling and Representation of Data
- Data Analysis
- Introduction to Natural Language Processing
- Data Engineering Systems
- Practicing Data Science Tools

University of Florida
Bachelor of Arts
Economics Major
Business Administration Minor
Pre-Med Track

Graduated:
December 2023

GPA:
3.90 / 4.00

Advanced Coursework:
- Biology
- Physics
- Organic Chemistry
- Microeconomics
- Macroeconomics
- Business Administration

Leadership & Campus Involvement:
- Director, Center for Undergraduate Research
- Advocacy Chair, Partners In Health
- General Chemistry Teaching Assistant
- International Research Director, Hearts for the Homeless
- Planned Parenthood Volunteer

Niceville High School

High School Diploma
Graduated: 2019

Activities & Awards:
- Speech and Debate Team Captain
- NAACP ACT-SO Bronze Medalist (Oratory)
- Florida Forensic League State Finalist in Original Oratory
"""

document_personality = """
Name: Simrun Sharma

Career Goals:
- Transitioning from Data Science into AI Engineering.

Target Roles:
- AI Engineer
- Applied AI Engineer
- AI Deployment Engineer
- Forward Deployed AI Engineer
- AI Solutions Engineer

Industries of Interest:
- Artificial Intelligence
- Healthcare Technology
- Defense Technology
- Data Science

Professional Interests:
- Artificial Intelligence
- Machine Learning
- Generative AI
- Agentic AI Systems
- Retrieval-Augmented Generation (RAG)
- Large Language Models (LLMs)
- Data Science
- Healthcare Analytics
- Brain Computer Interfaces
- Neurotechnology
- Explainable AI

Healthcare Interests:
- Passionate about applying AI and data science to improve patient outcomes, healthcare operations, accessibility, and clinical decision-making.

Learning Style:
- Step-by-step explanations
- Visual examples
- Interactive discussions
- Hands-on projects
- Building intuition before technical depth

Communication Style:
- Curious
- Analytical
- Direct
- Practical
- Detail-oriented

Work Preferences:
- Enjoys solving real-world problems.
- Likes collaborating with stakeholders.
- Prefers building practical AI systems with measurable impact.
- Enjoys end-to-end solution development.

Personality Traits:
- Curious
- Ambitious
- Persistent
- Analytical
- Detail-oriented
- Growth-oriented

Personal Interests:
- Pickleball
- Bachata
- Salsa
- Bollywood dance
- Strength training
- Fitness
- Pineapple on pizza
"""

#Chunking Function------------------
#Chunk the document
def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 150) -> list[str]:
    #Roughly, 1 token ≈ 4 characters in English text. 
    #Chunk size 1024 sand overlap of 150 characters is ideal based on document length or user needs look at RAG Ipynb
    
    BOUNDARIES = ["\n\n", "\n", ". ", " "]

    def find_natural_boundary(start: int, end: int) -> int:
        midpoint = start + (chunk_size // 2)
        for boundary in BOUNDARIES:
            pos = text.rfind(boundary, midpoint, end)
            if pos != -1:
                return pos + len(boundary)
        return end

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            end = find_natural_boundary(start, end)

        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks

#Generating chunks,ids, and metadatas--------------------------
#Generating chunks for multiple documents

#We want to have these documents as a list of dictionaries:
#Generating chunks for multiple documents

#We want to have these documents as a list of dictionaries:
documents = [
    {'text' : document_experience, 'source' : 'Simrun Sharma Experience'},
    {'text' : document_education, 'source' : 'Simrun Sharma Education'},
    {'text' : document_personality, 'source' : 'Simrun Sharma Personality'}   
]

#I want unique ids for each of the ids
# import uuid

#for each of the document chunk them, id them, metadata add to list:
ids = []
metadatas = []
chunks = []

for doc in documents:
    _chunks = chunk_text(text = doc['text'], chunk_size=300, overlap=30)
    _ids = [str(uuid.uuid4()) for i in range(len(_chunks))]
    _metadatas = [{'source' : doc['source'], 'chunk_index' : i} for i in range(len(_chunks))]

    #extend becase I am getting dictionaries I want to have one long list of dictionaries

    chunks.extend(_chunks)
    ids.extend(_ids)
    metadatas.extend(_metadatas)

#logs for debugging chunks 
print(f'The length of the chunks is {len(chunks)}')
print('\n')
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i + 1} | {len(chunk)} total characters ---")
    print(f'(IDS: {ids[i]}, SOURCE: {metadatas[i]['source']}, Chunk_index: {metadatas[i]['chunk_index']})')
    print('\n')
    print(chunk)

#Generate embeddings for chunks:

response = client.embeddings.create(
    model = 'text-embedding-3-small',
    input= chunks
)
#creating a list of embeddings above in response.data 
#I have the a list of Embedding objects and I just needs a list of the dimensions of each Embedding object
embedding  = [item.embedding for item in response.data]

#logs for embeddings:
print(f'The number of embeddings from the chunks: {len(embedding)}')
print(f'The number of dimensions is: {len(embedding[0])}')


#Initialize the ChromaDB and store the vectors
#Persistent Client will show in VSCode
chroma_client = chromadb.PersistentClient(path = "./digitaltwin_chroma")

#Creating Collection
collection = chroma_client.get_or_create_collection(name = 'Simrun_Information')


#If there is a collection already then we need to delete the collection
if collection.get()['ids']:
    collection.delete(collection.get()['ids'])

#Adding to our collection
collection.add(
    ids=ids,
    embeddings=embedding,
    metadatas=metadatas,
    documents=chunks
)


#System message---------------------
system_message = """
You are a digital twin of Simrun Sharma.
When people message you, you respond as Simrun - You answer questions as Simrun using first-person language. Using her voice, personality, knowledge, and kindness.

IMPORTANT RULES:

1. DO NOT use outside knowledge about Simrun.
2. DO NOT infer facts that are not explicitly stated.
3. DO NOT invent experiences, opinions, accomplishments, preferences, relationships, skills, or goals. Don't make up from internet.
4. The only factual information is given to you in this system message and anything you dont know you can respond saying:
"I don't know based on the information available to me."""

#Main Response Function------------

def respond_system_enhanced (message, history):
    #RAG
    response = client.embeddings.create(
        model = 'text-embedding-3-small',
        input = [message]
    )

    print(f' The length of dimensions of the message : {len(response.data[0].embedding)}')


    #we just want the first embedding closet distance
    query_embedding = response.data[0].embedding

    #searching through collection for chunks with closest distance to query embedding:

    results = collection.query(
        query_embeddings= [query_embedding],
        n_results= 3,
        include= ['documents','metadatas','distances']
    )
    
    #logs for debugging
    #lets just give the closest distance one as context
    context = "\n--\n".join(results['documents'][0])
    print(f'User message:\n{message}\n')
    print('Context this turn:\n', context)
    

    #system enhanced message:

    system_message_enhanced = system_message + "\n\n Context is:" + context


    #Build messages for each turn:
    messages = [{"role" : "system", "content": system_message_enhanced}] + history + [{"role" : "user", "content" : message}]

    # Call LLM
    response = client.chat.completions.create(
        model = "gpt-4.1-mini",
        messages = messages
    )

    
    message = response.choices[0].message

    return (message.content)




#launching gradio:
#inbrowser launch not needed
gr.ChatInterface(fn=respond_system_enhanced).launch()

