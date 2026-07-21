"""
Deploying Digital Twin: Built from Digital Twin IPYNB
"""
import os
from openai import OpenAI
from IPython.display import display, Markdown
import gradio as gr 
from dotenv import load_dotenv
import json
import uuid
import random
import requests
import chromadb
from pprint import pprint


#Setup-------------

#no loading the environment needed because its app.py
#You don't need it in production, hugging face already does it, but ipynb its to load into local machine
load_dotenv()

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

    #system enhanced message:

    system_enhanced_message = system_message + "\n\nContext:\n" + document_experience

    #logs for debugging
    print("\n===============\n")
    print("*****User message\n", message)
    print("\n****Context:\n", system_enhanced_message)

    #Build messages for each turn:
    messages = [{'role' : 'system', 'context':system_enhanced_message}] + history + [{'role':'user','context': message}]

    #calling the LLM:
    
    response = client.chat.completions.create(
        model = 'gpt-4.1-mini',
        messages=messages
    )

    #returning message content:
    message = response.choices[0].message

    return (message.content)

#launching gradio:
#inbrowser launch not needed
gr.ChatInterface(fn=respond_system_enhanced).launch()

