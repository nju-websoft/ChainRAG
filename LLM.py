import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
import logging

# Disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

RETRY_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

def retry_on_exception(func):
    return retry(
        stop=stop_after_attempt(5),  
        wait=wait_exponential(multiplier=1, min=4, max=10),  
        retry=retry_if_exception_type(RETRY_EXCEPTIONS),
        reraise=True
    )(func)
def clean_text(text: str) -> str:
    """Clean non-ASCII characters from text"""
    # Replace common special quotes and punctuation
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...',
        '•': '*',
        '°': ' degrees ',
        '×': 'x',
        '÷': '/',
        '≠': '!=',
        '≤': '<=',
        '≥': '>=',
        '±': '+/-',
        '∞': 'infinity',
        '′': "'",
        '″': '"',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '©': '(c)',
        '®': '(R)',
        '™': '(TM)',
    }
    
    # Apply replacements
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove remaining non-ASCII characters
    cleaned_text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Clean extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text
@retry_on_exception
def custom_embedding(texts,api_key):
    # Clean and check input text
    cleaned_texts = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
        # Clean text
        cleaned_text = clean_text(text)
        cleaned_texts.append(cleaned_text)
    batch_size = 20
    client = OpenAI(api_key)
    
    texts = [str(text).strip() for text in cleaned_texts]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_texts,
                encoding_format="float"
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"处理批次 {i} 时出错: {str(e)}")
            empty_embedding = [0.0] * 1536
            embeddings.extend([empty_embedding] * len(batch_texts))
    return embeddings

@retry_on_exception
def custom_llm(prompt, api_key):
    client = OpenAI(api_key)
    if isinstance(prompt, dict):
        formatted_prompt = prompt.get("text", "")
    elif isinstance(prompt, str):
        formatted_prompt = prompt
    else:
        formatted_prompt = str(prompt)
    completion = client.chat.completions.create(
    model="gpt-4o-mini", # 
    messages=[
        {'role': 'user', 'content': formatted_prompt}],
    temperature=0.2,
    )
    return completion.choices[0].message.content

class CustomEmbeddings:
    def __init__(self, api_key):
        self.api_key = api_key

    def embed_documents(self, texts):
        return custom_embedding(texts, self.api_key)

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def __call__(self, text):
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or a list of strings") 