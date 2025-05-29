import spacy
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
import requests
from tenacity import retry
import time
import json
import re
from typing import List, Optional, Tuple, Set
from functools import wraps
from LLM import custom_llm,  CustomEmbeddings
from utils import save_embeddings, load_embeddings
from tqdm import tqdm

RETRY_EXCEPTIONS = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)

def timeout_handler(signum, frame):
    """Handle timeouts in API requests"""
    raise TimeoutError("Request timeout")

model_path = "en_core_web_sm"
nlp = spacy.load(model_path)

def split_into_sentences(text: str) -> List[str]:
    """
    Split input text into individual sentences using spaCy.
    
    Args:
        text: Input text to be split
        
    Returns:
        List of sentences
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences

def compute_similarity_matrix(sentence_embeddings):
    """
    Compute cosine similarity matrix between sentence embeddings.
    
    Args:
        sentence_embeddings: List of sentence embedding vectors
        
    Returns:
        Similarity matrix where each element [i,j] represents similarity between sentences i and j
    """
    embeddings = np.array(sentence_embeddings)
    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
    return similarity_matrix

def calculate_bm25_importance(sentences, ent_lists):
    """
    Calculate entity importance scores using BM25 algorithm.
    
    Args:
        sentences: List of sentences
        ent_lists: List of entity lists for each sentence
        
    Returns:
        Tuple of (high_importance_entities, entity_scores)
    """
    entity_docs = []
    all_entities = set()
    
    # Extract entities from each sentence
    for ent_list in ent_lists:
        entities = [ent[0] for ent in ent_list]
        entity_docs.append(entities)
        all_entities.update(entities)
    
    # Create BM25 model from entity documents
    bm25 = BM25Okapi(entity_docs)

    # Calculate importance score for each entity
    entity_scores = {}
    for entity in all_entities:
        scores = bm25.get_scores([entity])
        avg_score = np.mean(scores)
        entity_scores[entity] = avg_score
    
    # Identify high importance entities (above 40th percentile)
    scores = list(entity_scores.values())
    threshold = np.percentile(scores, 40)  
    high_importance_entities = {ent for ent, score in entity_scores.items() 
                              if score > threshold}
    
    return high_importance_entities, entity_scores

def build_sentence_graph(sentences, similarity_matrix, ent_lists, entity_scores, k=10):
    """
    Build a graph where nodes are sentences and edges represent different relationships.
    
    Args:
        sentences: List of sentences
        similarity_matrix: Similarity matrix between sentences
        ent_lists: List of entity lists for each sentence
        entity_scores: Dictionary mapping entities to importance scores
        k: Number of similarity edges per sentence
        
    Returns:
        NetworkX graph with sentence relationships
    """
    G = nx.Graph()
    n = len(sentences)
    
    k = min(k, n-1)
    
    # Create set of entities for each sentence
    sentence_ents = [{ent[0] for ent in ent_list} for ent_list in ent_lists]
    
    # Add nodes (sentences)
    G.add_nodes_from(range(n))
    
    # Add similarity edges (top-k most similar sentences)
    edges_similarity = []
    for i in range(n):
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1  # Exclude self-similarity
        top_k_indices = np.argpartition(similarities, -k)[-k:]
        edges_similarity.extend((i, j, {
            'weight': similarities[j],
            'label': 'similarity'
        }) for j in top_k_indices)
    G.add_edges_from(edges_similarity)
    
    # Add positional edges (nearby sentences in text)
    edges_position = []
    for i in range(n):
        for j in range(max(0, i-3), min(n, i+4)):
            if i != j:
                edges_position.append((i, j, {
                    'weight': 1.0 / (abs(i-j) + 1),
                    'label': 'position'
                }))
    G.add_edges_from(edges_position)
    
    # Add entity-based edges (sentences sharing important entities)
    edges_entity = []
    for i in range(n):
        for j in range(i+1, n):
            common_ents = sentence_ents[i] & sentence_ents[j]
            if common_ents:
                weight = sum(entity_scores.get(ent, 0) for ent in common_ents)
                edges_entity.append((i, j, {
                    'weight': weight,
                    'label': 'entity',
                    'entities': list(common_ents)
                }))
    G.add_edges_from(edges_entity)
    
    return G

def construct_entity_graph(sentences: List[str], API_KEY: str, dataset_name: str = "musique", text_id: str = "default", use_cache: bool = True) -> Tuple[List[np.ndarray], nx.Graph, Set[str]]:
    """
    Main function to construct the sentence graph with entity information.
    
    Args:
        sentences: List of sentences
        API_KEY: API key for embedding model
        dataset_name: Name of dataset
        text_id: Unique identifier for the text
        use_cache: Whether to use cached embeddings
        
    Returns:
        Tuple of (sentence_embeddings, sentence_graph, high_importance_entities)
    """
    embeddings = CustomEmbeddings(API_KEY)

    # Extract entities from all sentences
    ent_lists = []
    # Process all sentences at once for efficiency
    docs = list(nlp.pipe(sentences))
    for doc in docs:
        ents = [(ent.text, ent.label_) for ent in doc.ents]
        ent_lists.append(ents)

    # Load or compute embeddings
    cached_embeddings = None
    if use_cache:
        cached_embeddings = load_embeddings(dataset_name, text_id)
        
    if cached_embeddings is not None:
        sentence_embeddings = cached_embeddings
    else:
        sentence_embeddings = [np.array(emb) for emb in embeddings.embed_documents(sentences)]
        if use_cache:
            save_embeddings(sentence_embeddings, dataset_name, text_id)
    
    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(sentence_embeddings)

    # Calculate entity importance
    high_importance_entities, entity_scores = calculate_bm25_importance(sentences, ent_lists)
    
    # Build sentence graph
    time_start = time.time()
    sentence_graph = build_sentence_graph(sentences, similarity_matrix, ent_lists, entity_scores)
    
    return sentence_embeddings, sentence_graph, high_importance_entities


def decompose_question(question: str, api_key: str) -> List[str]:
    """
    Decompose a complex question into multiple simpler sub-questions.
    
    This function first determines if a question requires multi-hop reasoning,
    then breaks it down into sub-questions if needed.
    
    Args:
        question: Original question
        api_key: API key for LLM
        
    Returns:
        List of sub-questions (or single question if decomposition isn't needed)
    """
    try:
        # First, determine if question is multi-hop
        judge_prompt = """You are a helpful AI assistant that determines if a question requires multiple steps to answer.

        Guidelines for identifying multi-hop questions:
        1. The question requires finding and connecting multiple pieces of information
        2. The answer cannot be found in a single direct statement
        3. You need to find intermediate information to reach the final answer

        Output format should be a JSON object with only one fields:
        - "is_multi_hop": boolean (true/false)

        Example:
        Question: "Who is the paternal grandmother of Marie Of Brabant, Queen Of France?"
        Output: {"is_multi_hop": false}
        Question: "Who is Archibald Acheson, 4Th Earl Of Gosford's paternal grandfather?"
        Output: {"is_multi_hop": false}
        Question: "Who was the wife of the person who founded Microsoft?"
        Output: {"is_multi_hop": true}"""

        judge_full_prompt = f"{judge_prompt}\n\nQuestion: {question}\n\nIs this a multi-hop question?"

        try:
            response = custom_llm(judge_full_prompt, api_key)
            cleaned_response = response.strip()
            
            # Extract JSON part from response
            if '```json' in cleaned_response:
                cleaned_response = cleaned_response.split('```json')[1].split('```')[0]
            elif '```' in cleaned_response:
                cleaned_response = cleaned_response.split('```')[1].split('```')[0]
            
            cleaned_response = cleaned_response.strip()
            
            # Try to fix common JSON format issues
            if not cleaned_response.startswith('{'):
                cleaned_response = '{' + cleaned_response.split('{', 1)[1]
            if not cleaned_response.endswith('}'):
                cleaned_response = cleaned_response.rsplit('}', 1)[0] + '}'
            
            try:
                judgment = json.loads(cleaned_response)
            except json.JSONDecodeError:
                # Try more aggressive cleaning for JSON parsing
                cleaned_response = re.sub(r'[^\{\}\:\"\,\w\.\-\_\s]', '', cleaned_response)
                try:
                    judgment = json.loads(cleaned_response)
                except:
                    print(f"Cannot parse JSON: {cleaned_response}")
                    return [question]
            
            # If not multi-hop, return original question
            if not judgment.get('is_multi_hop', True):  
                return [question]

        except Exception as e:
            print(f"Error in question type classification: {str(e)}")
            return [question]

        # For multi-hop questions, decompose into sub-questions
        system_prompt = """You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions.    
        Guidelines:
        1. Only break down the question if it requires finding and connecting multiple distinct pieces of information
        2. Each sub-question should target a specific, essential piece of information
        3. Avoid generating redundant or overlapping sub-questions
        4. For questions about impact/significance, focus on:
        - What was the thing/event
        - What was its impact/significance
        5. For comparison questions between two items (A vs B):
        - First identify the specific attribute being compared for each item
        - Then ask about that attribute for each item separately
        - For complex comparisons, add a final question to compare the findings
        6.**Logical Progression**:
        Sub-questions should have clear relationships, such as:
        - **Parallel**: Independent sub-questions that both contribute to answering the original question.
        Example:
        Original: "What are the causes and consequences of climate change on global ecosystems?"
        Output: ["What are the main causes of climate change?", "What are the major consequences of climate change on global ecosystems?"]
        - **Sequential**: Sub-questions that build upon each other step-by-step.
        Example:
        Original: "What university, founded in 1890, is known for its groundbreaking work in economics?"
        Output: ["Which universities were founded in 1890?", "Which of these universities is known for its groundbreaking work in economics?"]
        - **Comparative**: Questions that compare attributes between items.
        Example 1:
        Original: "Which film has the director who was born earlier, The Secret Invasion or The House Of The Seven Hawks?"
        Output: ["Who directed The Secret Invasion and when was this director born?", "Who directed The House Of The Seven Hawks and when was this director born?"]
        Example 2:
        Original: "Do both films The Reincarnation Of Golden Lotus and I'll Get By (Film) have directors from the same country?"
        Output: ["Who directed The Reincarnation Of Golden Lotus and which country is he/she from?", "Who directed I'll Get By (Film) and which country is he/she from?"]

        7. Keep the total number of sub-questions minimal (usually 2 at most)

        Output format should be a JSON array of sub-questions. For example:
        Original: "Were the wireless earbuds Apple introduced in 2016 revolutionary for the market?"
        Output: ["What wireless earbuds did Apple introduce in 2016?", "How did these earbuds impact the wireless earbud market?"]

        Remember: Each sub-question must be necessary and distinct. Do not create redundant questions. For comparison questions, focus on gathering the specific information needed for the comparison in the most efficient way."""
        
        full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nBreak down this question into minimal necessary sub-questions:"
        
        try:
            response = custom_llm(full_prompt, api_key)
            cleaned_response = response.strip()
            
            # Extract JSON part
            if '```json' in cleaned_response:
                cleaned_response = cleaned_response.split('```json')[1].split('```')[0]
            elif '```' in cleaned_response:
                cleaned_response = cleaned_response.split('```')[1].split('```')[0]
            
            cleaned_response = cleaned_response.strip()
            
            # Try to parse the JSON response
            try:
                sub_questions = json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"Original JSON parsing error: {str(e)}")
                print(f"Attempting to clean and reparse: {cleaned_response}")
                
                # Handle malformed JSON list format
                if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                    # Simple string splitting for lists
                    items = cleaned_response[1:-1].split('","')
                    sub_questions = [item.strip('"\'') for item in items]
                else:
                    print("Cannot parse as valid JSON array, using original question")
                    return [question]
            
            if isinstance(sub_questions, list) and sub_questions:
                return sub_questions
            else:
                print("API did not return a valid list of questions")
                return [question]
                
        except Exception as e:
            print(f"Error processing sub-questions: {str(e)}")
            return [question]
            
    except Exception as e:
        print(f"Error in question decomposition process: {str(e)}")
        return [question]


