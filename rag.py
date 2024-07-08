import boto3
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import LlamaTokenizer

sage_client = boto3.client('sagemaker-runtime')
endpoint_name = os.environ.get('PREDICTOR_ENDPOINT')

def query_document_retrieval_system(query, documents_dir='./source_documents', top_k=2):
    """
    Retrieve the top-k most relevant documents based on the query using TF-IDF.
    """
    documents = []
    filenames = []

    # Load documents from the specified directory
    for filename in os.listdir(documents_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(documents_dir, filename), 'r') as file:
                documents.append(file.read())
                filenames.append(filename)

    corpus = documents + [query]
    
    vectorizer_tfidf = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer_tfidf.toarray()
    
    query_vector = vectors[-1]  # Last vector is the query
    doc_vectors = vectors[:-1]  # Others - documents
    cosine_similarities = cosine_similarity([query_vector], doc_vectors).flatten()

    # Get top-k similar docs
    top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
    top_k_documents = [documents[i] for i in top_k_indices]

    return top_k_documents

def generate_response(query, endpoint_name):
    documents = query_document_retrieval_system(query)

    combined_input = query + " ".join(documents)

    # Tokenize
    tokenizer = LlamaTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
    tokenized_input = tokenizer.encode(combined_input, return_tensors="pt").tolist()

    payload = { 'inputs': tokenized_input }

    # Invoke the SageMaker endpoint
    response = sage_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(payload)
    )

    # Decode the response
    response_body = response['Body'].read().decode('utf-8')
    response_dict = json.loads(response_body)
    generated_text = tokenizer.decode(response_dict[0], skip_special_tokens=True)

    return generated_text


def chatbot():
    print("Chatbot initialized. Type 'exit' to quit.")
    print("Example usage: \nquery: How to proceed with a refund for a product A if I'm from UAE?")
    while True:
        query = input("query: ")
        if query.lower() == 'exit':
            print("Chatbot session ended.")
            break
        response = generate_response(query, endpoint_name)
        print(f"Bot: {response}")

if __name__ == "__main__":
  chatbot()
