from typing import List, Dict, Any, Tuple, Set
import numpy as np
import networkx as nx
from sentence_graph import (
    construct_entity_graph, 
    split_into_sentences,
    decompose_question,
)
from LLM import custom_llm, CustomEmbeddings
import torch
import json
import argparse
from tqdm import tqdm
import time
from FlagEmbedding import FlagReranker
import os
from pathlib import Path
from utils import (
    load_dataset, 
    append_result, 
    DATASETS,
    logger
)

# Global configuration
API_KEY = "your_api_key"  # Configure your API key here

# Configure GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
torch.cuda.set_device(0)  

# Initialize reranker
try:
    reranker = FlagReranker(
    model_name_or_path="BAAI/bge-reranker-large",
    use_fp16=True # Set to False when using CPU
)
    print("Reranker initialized successfully")
except Exception as e:
    print(f"Error initializing reranker: {str(e)}")
    reranker = None

# Global counters for statistics
llm_counter = 0
context_length_total = 0
processing_time_total = 0
total_questions = 0

class QuestionPlanner:
    """
    Main class for planning and executing multi-hop question answering.
    Handles sentence graph construction, question decomposition, and context retrieval.
    """
    
    def __init__(self, api_key: str, dataset_name: str, use_cache: bool = True):
        """
        Initialize the QuestionPlanner with API key and dataset configuration.
        
        Args:
            api_key: API key for LLM and embedding models
            dataset_name: Name of the dataset being used
            use_cache: Whether to use cached embeddings
        """
        self.api_key = api_key
        self.embeddings = CustomEmbeddings(api_key)
        self.dataset_name = dataset_name
        self.use_cache = use_cache
        self.clear_graph()
        print(f"Initializing QuestionPlanner, dataset: {dataset_name}, cache: {'enabled' if use_cache else 'disabled'}")
    
    def clear_graph(self):
        """Reset all graph-related data structures"""
        self.similarity_matrix = None
        self.sentence_graph = None
        self.high_tfidf_entities = None
        self.sentences = None
        self.sentence_embeddings = None
    
    def build_graph(self, text: str, text_id: str = "default") -> None:
        """
        Build a sentence graph from the given text.
        
        Args:
            text: Input text document
            text_id: Unique identifier for caching
        """
        self.sentences = split_into_sentences(text)
        self.sentence_embeddings, self.sentence_graph, self.high_tfidf_entities = construct_entity_graph(
            self.sentences,
            self.api_key,
            dataset_name=self.dataset_name,
            text_id=text_id,
            use_cache=self.use_cache
        )

    def get_n_hop_neighbors(self, sentence: str, n_hops: int = 1) -> Set[str]:
        """
        Get sentences that are n-hops away from the given sentence in the graph.
        
        Args:
            sentence: Source sentence
            n_hops: Maximum number of hops in the graph
            
        Returns:
            Set of neighboring sentences
        """
        try:
            sent_idx = self.sentences.index(sentence)
            neighbors = set()
            
            # Get subgraph containing nodes within n_hops of the source
            ego_graph = nx.ego_graph(self.sentence_graph, sent_idx, radius=n_hops)
            
            # Collect all sentences in the subgraph (excluding the source)
            for node in ego_graph.nodes():
                if node != sent_idx:
                    neighbors.add(self.sentences[node])
                    
            return neighbors
            
        except Exception as e:
            print(f"Error getting neighbor sentences: {str(e)}")
            return set()
    
    def analyze_with_context(self, sub_question: str, context_sentences: List[str]) -> Dict[str, Any]:
        """
        Create an analysis object with a sub-question and its context.
        
        Args:
            sub_question: The sub-question
            context_sentences: Retrieved context sentences
            
        Returns:
            Dictionary with sub-question and retrieved context
        """
        return {
            "sub_question": sub_question,
            "retrieved_context": context_sentences
        }
    
    def force_answer(self, sub_question: str, context_sentences: List[str]) -> str:
        """
        Generate a concise answer to a sub-question based on provided context.
        
        Args:
            sub_question: Question to answer
            context_sentences: Context information for answering
            
        Returns:
            Generated answer as a string
        """
        prompt = """Based on the given context, you must provide an answer with fewest words to the question.
        Only give me the answer and do not output any other words.
        
        Context: {context}
        Question: {question}
        
        Provide your best possible answer:"""
        
        try:
            global llm_counter
            llm_counter += 1
            response = custom_llm(prompt.format(
                context=" ".join(context_sentences),
                question=sub_question
            ), self.api_key)
            return response.strip()
            
        except Exception as e:
            print(f"Error forcing answer: {str(e)}")
            return "Unable to provide an answer due to error"
    
    def answer_original_question(self, question: str, sub_results: List[Dict]) -> Dict[str, str]:
        """
        Generate a final answer to the original question using two methods:
        1) Using sub-question answers
        2) Using all retrieved context directly
        
        Args:
            question: Original question
            sub_results: List of sub-question results with answers and context
            
        Returns:
            Dictionary with two answer versions
        """
        # Prompt for answering using sub-question results
        prompt_with_subq = """Based on the answers to the sub-questions, use the fewest words possible to answer the original question.
        Only give me the answer and do not output any other words.
        Original Question: {original_question}
        Sub-questions and their answers:
        {sub_answers}
        Provide the shortest possible answer:"""
        
        # Prepare text for sub-question answers
        sub_answers_text = "\n".join([
            f"Sub-question: {result['sub_question']}\nAnswer: {result['answer']}"
            for result in sub_results
        ])
        
        # Collect and deduplicate all context sentences
        all_contexts = []
        for result in sub_results:
            all_contexts.extend(result['context'])
        
        unique_contexts = list(dict.fromkeys(all_contexts))
        sorted_contexts = self.sort_context(question, unique_contexts)
    
        # Prompt for answering using all context directly
        prompt_with_context = """Based on the following context, use the fewest words possible to answer the original question.
        Only give me the answer and do not output any other words.
        Context: {context}
        Question: {question}
        
        Answer:"""
        
        try:
            global llm_counter
            llm_counter += 2  # Count both LLM calls
            
            # Generate answer using sub-question answers
            answer_with_subq = custom_llm(prompt_with_subq.format(
                original_question=question,
                sub_answers=sub_answers_text
            ), self.api_key).strip()
            
            # Generate answer using combined context
            answer_with_context = custom_llm(prompt_with_context.format(
                context="\n".join(sorted_contexts),
                question=question
            ), self.api_key).strip()
            
            return {
                "answer_with_subquestions": answer_with_subq,
                "answer_without_subquestions": answer_with_context,
            }
            
        except Exception as e:
            print(f"Error generating final answer: {str(e)}")
            return {
                "answer_with_subquestions": "Unable to provide an answer due to error",
                "answer_without_subquestions": "Unable to provide an answer due to error",
            }
            
    def sort_context(self, question: str, context_sentences: List[str]) -> List[str]:
        """
        Sort context sentences by relevance to the question using reranker.
        
        Args:
            question: Query question
            context_sentences: List of context sentences to sort
            
        Returns:
            Sorted list of context sentences by relevance
        """
        if not context_sentences:
            return []
            
        try:
            if reranker is None:
                return context_sentences
                
            # Create pairs of [question, sentence] for reranking
            pairs = [[question, sent] for sent in context_sentences]
            scores = reranker.compute_score(pairs)
            
            # Sort by relevance score (descending)
            scored_sentences = list(zip(scores, context_sentences))
            scored_sentences.sort(reverse=True)
            return [sent for _, sent in scored_sentences]
        except Exception as e:
            print(f"Error sorting context: {str(e)}")
            return context_sentences
            
    def plan(self, text: str, question: str) -> Dict[str, Any]:
        """
        Main planning function for answering a question.
        Decomposes question, retrieves context, and generates answers.
        
        Args:
            text: Document text
            question: Original question
            
        Returns:
            Dictionary with detailed results including sub-questions, context, and answers
        """
        # Build graph if not already built
        if self.sentence_embeddings is None:
            self.build_graph(text)
        
        global llm_counter, context_length_total  
        llm_counter += 1  # Count decomposition call
        
        # Decompose the question into sub-questions
        sub_questions = decompose_question(question, self.api_key)
        sub_results = []
        previous_answers = {}  
        previous_context_summary = None  
        
        # Process each sub-question sequentially
        for i, sub_q in enumerate(sub_questions):
            try:
                modified_sub_q = sub_q
                
                # Check if sub-question contains references to previous answers
                for prev_q, prev_ans in previous_answers.items():
                    if any(ref in sub_q.lower() for ref in ["this", "that", "the", "these", "those", "it", "they", "he", "she","his","her","its","their"]):
                        # Ask LLM if this question refers to previous answers
                        should_replace = custom_llm(f"""
                        Previous question: {prev_q}
                        Previous answer: {prev_ans}
                        Current question: {sub_q}
                        Does the current question refer to the answer of the previous question? Answer yes or no.
                        """, self.api_key).strip().lower()
                        
                        llm_counter += 1
                        if should_replace == "yes":
                            # Rewrite question to be self-contained
                            modified_sub_q = custom_llm(f"""
                            Rewrite the following question to be self-contained by replacing pronouns or references with the actual entities they refer to.
                            Previous question: {prev_q}
                            Previous answer: {prev_ans}
                            Current question: {sub_q}
                            Rewritten question:
                            """, self.api_key).strip()
                            llm_counter += 1
                            break
                
                # Retrieve initial context for the sub-question
                top_sentences = self.find_top_k_sentences(modified_sub_q, k=7)
                
                context = [sent for _, sent in top_sentences]
                context_length_total += len(context)
                
                # Add previous context if question refers to previous information
                prev_ans_context = []
                if previous_context_summary and any(ref in sub_q.lower() for ref in ["previous", "before", "earlier", "last", "first"]):
                    prev_ans_context = [previous_context_summary]
                
                # Check if current context is sufficient to answer the question
                can_answer = self.can_answer_question(modified_sub_q, context + prev_ans_context)
                
                # If not, expand context with graph neighbors
                if not can_answer:
                    context_length_total += len(prev_ans_context)
                    extended_context = []
                    for _, sent in top_sentences[:3]:
                        neighbors = self.get_n_hop_neighbors(sent, n_hops=1)
                        extended_context.extend(neighbors)
                    
                    if extended_context:
                        sorted_extended = self.sort_context(modified_sub_q, extended_context)
                        context.extend(sorted_extended[:5])
                        context_length_total += 5
                
                # Generate answer for this sub-question
                answer = self.force_answer(modified_sub_q, context + prev_ans_context)
                
                # Store results for this sub-question
                sub_result = {
                    "sub_question": sub_q,
                    "modified_sub_question": modified_sub_q,
                    "context": context + prev_ans_context,
                    "answer": answer
                }
                
                sub_results.append(sub_result)
                previous_answers[sub_q] = answer
                
                # Update context summary for next sub-question
                if not previous_context_summary:
                    previous_context_summary = f"For the question '{sub_q}', the answer is: {answer}"
                else:
                    previous_context_summary += f" For the question '{sub_q}', the answer is: {answer}"
                
            except Exception as e:
                print(f"Error processing sub-question: {str(e)}")
                sub_results.append({
                    "sub_question": sub_q,
                    "modified_sub_question": sub_q,
                    "context": [],
                    "answer": "Unable to process this sub-question due to error"
                })
        
        # Generate final answers to the original question
        final_answers = self.answer_original_question(question, sub_results)
        
        return {
            "original_question": question,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
            "final_answer_with_subquestions": final_answers["answer_with_subquestions"],
            "final_answer_without_subquestions": final_answers["answer_without_subquestions"]
        }
        
    def sort_section(self, question: str, sentences: List[Tuple[float, str]], k: int) -> List[Tuple[float, str]]:
        """
        Sort sentences by relevance to the question using reranker.
        
        Args:
            question: Query question
            sentences: List of (score, sentence) tuples
            k: Number of top sentences to return
            
        Returns:
            List of top-k (score, sentence) tuples sorted by relevance
        """
        try:
            start_time = time.time()
            pairs = [[question, sent] for _, sent in sentences]
            scores = reranker.compute_score(pairs)
            scored_sentences = [(score, sent) for score, (_, sent) in zip(scores, sentences)]
            scored_sentences.sort(reverse=True)
            return scored_sentences[:k]
        except KeyboardInterrupt:
            if hasattr(reranker, 'stop_self_pool'):
                reranker.stop_self_pool()
            raise
        except Exception as e:
            print(f"Error sorting sentences: {str(e)}")
            return sentences[:k]

    def find_top_k_sentences(self, question: str, k: int) -> List[Tuple[float, str]]:
        """
        Find the top-k most relevant sentences for a question.
        Uses embedding similarity followed by reranker for better results.
        
        Args:
            question: Query question
            k: Number of sentences to retrieve
            
        Returns:
            List of (score, sentence) tuples for top-k sentences
        """
        # Get embedding for the question
        q_embedding = np.array(self.embeddings.embed_query(question))
        q_norm = np.linalg.norm(q_embedding)
        
        if q_norm == 0:
            print(f"Warning: Query vector norm is zero: {question}")
            return [(0, s) for s in self.sentences[:k]]
        
        # Calculate similarity between question and all sentences
        sentence_scores = []
        for sent_idx, sent_embedding in enumerate(self.sentence_embeddings):
            try:
                sent_norm = np.linalg.norm(sent_embedding)
                if sent_norm == 0:
                    similarity = 0
                else:
                    # Cosine similarity calculation
                    similarity = np.dot(q_embedding, sent_embedding) / (q_norm * sent_norm)
                
                sentence = self.sentences[sent_idx]
                sentence_scores.append((similarity, sentence))
            except Exception as e:
                print(f"Error processing sentence: {str(e)}")
                sentence_scores.append((0, self.sentences[sent_idx]))
        
        # Sort by similarity score (first round)
        sentence_scores.sort(reverse=True)
        k1 = min(100, len(sentence_scores))
        top_100_sentences = sentence_scores[:k1]
        
        # Rerank using neural reranker (second round)
        return self.sort_section(question, top_100_sentences, k)

    def can_answer_question(self, question: str, context_sentences: List[str]) -> bool:
        """
        Check if the given context contains enough information to answer the question.
        
        Args:
            question: Query question
            context_sentences: List of context sentences
            
        Returns:
            Boolean indicating whether the question can be answered
        """
        prompt = """Based on the following context, can you answer the question? 
        Please respond with 'yes' or 'no' only.
        
        Question: {question}
        Context: {context}
        
        Can you answer the question based on this context? (yes/no):"""
        
        try:
            response = custom_llm(prompt.format(
                question=question,
                context="\n".join(context_sentences)
            ), self.api_key).strip().lower()
            
            return "yes" in response
            
        except Exception as e:
            print(f"Error checking if question can be answered: {str(e)}")
            return False


def chain_rag(dataset_name: str, use_cache: bool = True):
    """
    Execute ChainRAG algorithm
    
    Args:
        dataset_name: Dataset name
        use_cache: Whether to use cache
    """
    global llm_counter, API_KEY, total_questions, context_length_total, processing_time_total
    
    # Setup output directories
    output_dir = Path("processed_data") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "results.jsonl"
    
    # Initialize planner and load dataset
    planner = QuestionPlanner(API_KEY, dataset_name, use_cache)
    data = load_dataset(dataset_name)
    total_questions = len(data)
    
    # Process each question with progress bar
    progress_bar = tqdm(data, desc="Processing questions", unit="question")
    for idx, item in enumerate(progress_bar):
        question = item["question"]
        context = item["context"]
        expected_answer = item["expected_answer"]
        
        text_id = f"question_{idx}"
        
        progress_bar.set_description(f"Question {idx+1}/{len(data)}")
        
        start_time = time.time()
        
        # Reset counter for this question
        question_llm_counter = 0
        question_context_length = 0
        
        try:
            # Build graph and plan for this question
            planner.clear_graph()
            planner.build_graph(context, text_id)
            
            # Record initial counter
            initial_llm_counter = llm_counter
            
            # Execute the multi-hop reasoning process
            result = planner.plan(context, question)
            
            # Calculate statistics for this question
            question_llm_counter = llm_counter - initial_llm_counter
            
            # Calculate context length
            for sub_result in result["sub_results"]:
                question_context_length += len(sub_result["context"])
            
            end_time = time.time()
            question_processing_time = end_time - start_time
            
            # Update totals
            processing_time_total += question_processing_time
            
            # Update progress bar with statistics
            progress_bar.set_postfix({
                "LLM calls": question_llm_counter, 
                "Context len": question_context_length,
                "Time": f"{question_processing_time:.2f}s"
            })
            
            # Prepare output for this question
            output = {
                "question_id": idx,
                "question": question,
                "expected_answer": expected_answer,
                "answer_with_subquestions": result["final_answer_with_subquestions"],
                "answer_without_subquestions": result["final_answer_without_subquestions"],
                "sub_questions": result["sub_questions"],
                "total_context_words": question_context_length,
                "llm_calls": question_llm_counter,
                "processing_time": question_processing_time
            }
            
            # Save results
            append_result(output, str(output_file))
            
        except Exception as e:
            print(f"Error processing question {idx}: {str(e)}")
            
            end_time = time.time()
            question_processing_time = end_time - start_time
            processing_time_total += question_processing_time
            
            # Save error information
            error_output = {
                "question_id": idx,
                "question": question,
                "expected_answer": expected_answer,
                "error": str(e),
                "llm_calls": llm_counter - initial_llm_counter,
                "processing_time": question_processing_time
            }
            
            append_result(error_output, str(output_file))
    
    # Print summary statistics
    if total_questions > 0:
        avg_llm_calls = llm_counter / total_questions
        avg_context_length = context_length_total / total_questions
        avg_processing_time = processing_time_total / total_questions
        
        print("\n===== PROCESSING SUMMARY =====")
        print(f"Total questions processed: {total_questions}")
        print(f"Average LLM calls per question: {avg_llm_calls:.2f}")
        print(f"Average context length per question: {avg_context_length:.2f}")
        print(f"Average processing time per question: {avg_processing_time:.2f} seconds")
        print("===============================")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ChainRAG algorithm on specified dataset")
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="musique",
        choices=list(DATASETS.keys()),
        help="Name of dataset to use"
    )
    
    parser.add_argument(
        "--no_cache", 
        action="store_true",
        help="Disable cache (default: cache enabled)"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    chain_rag(
        dataset_name=args.dataset,
        use_cache=not args.no_cache
    )
