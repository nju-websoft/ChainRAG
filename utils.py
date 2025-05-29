import os
import pickle
import faiss
import numpy as np
import json
import logging
from typing import List, Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chain_rag.log", mode="a")
    ]
)
logger = logging.getLogger("chain_rag")

# Dataset configuration
DATASETS = {
    "hotpotqa": {
        "path": "data/hotpotqa.jsonl",
    },
    "musique": {
        "path": "data/musique.jsonl",
    },
    "2wikimqa": {
        "path": "data/2wikimqa.jsonl",
    }
}

def get_embeddings_cache_path(dataset_name: str, text_id: str) -> str:
    """
    Get the path for the embeddings cache file
    
    Args:
        dataset_name: Name of the dataset
        text_id: Text ID
        
    Returns:
        Path to the embeddings cache file (without extension)
    """
    cache_dir = f"cache/{dataset_name}/embeddings"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"embeddings_{text_id}")

def save_embeddings(embeddings: List[np.ndarray], dataset_name: str, text_id: str):
    """
    Save embeddings to cache
    
    Args:
        embeddings: List of embedding vectors
        dataset_name: Name of the dataset
        text_id: Text ID
    """
    embeddings_array = np.array(embeddings).astype('float32')
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension) 
    index.add(embeddings_array)
    
    cache_path = get_embeddings_cache_path(dataset_name, text_id)
    faiss.write_index(index, f"{cache_path}.index")
    with open(f"{cache_path}.pkl", 'wb') as f:
        pickle.dump(embeddings_array, f)

def load_embeddings(dataset_name: str, text_id: str) -> Optional[List[np.ndarray]]:
    """
    Load embeddings from cache
    
    Args:
        dataset_name: Name of the dataset
        text_id: Text ID
        
    Returns:
        List of embedding vectors, or None if cache doesn't exist
    """
    cache_path = get_embeddings_cache_path(dataset_name, text_id)
    if os.path.exists(f"{cache_path}.index") and os.path.exists(f"{cache_path}.pkl"):
        try:
            with open(f"{cache_path}.pkl", 'rb') as f:
                embeddings_array = pickle.load(f)
            return list(embeddings_array)
        except Exception as e:
            print(f"Error loading cached embeddings: {str(e)}")
            return None
    return None

def load_dataset(dataset_name: str, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load the specified dataset
    
    Args:
        dataset_name: Name of the dataset
        num_samples: Number of samples to load, if None load all
        
    Returns:
        List containing the processed data
    """
    if dataset_name not in DATASETS:
        available_datasets = ", ".join(DATASETS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {available_datasets}")
    
    dataset_path = DATASETS[dataset_name]["path"]
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    logger.info(f"Loading dataset: {dataset_name}")
    
    processed_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples is not None and i >= num_samples:
                    break
                    
                data = json.loads(line.strip())
                
                # Process different dataset formats
                if dataset_name in ["musique", "2wikimqa"]:
                    processed_item = {
                        'question': data.get('input', ''),
                        'context': data.get('context', ''),
                        'expected_answer': data.get('answers', '')
                    }
                else:
                    # Default format
                    processed_item = {
                        'question': data.get('question', data.get('input', '')),
                        'context': data.get('context', ''),
                        'expected_answer': data.get('answer', data.get('answers', ''))
                    }
                
                processed_data.append(processed_item)
        
        logger.info(f"Successfully loaded {len(processed_data)} samples")
        return processed_data
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise


def append_result(result: Dict[str, Any], output_file: str) -> None:
    """
    Append a single result to file
    
    Args:
        result: Single result dictionary
        output_file: Output file path
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False,indent=4)
            f.write("\n")
    except Exception as e:
        print(f"Error appending result: {str(e)}") 