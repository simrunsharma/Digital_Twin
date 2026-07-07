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
