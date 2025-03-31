import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase and removing special characters.
    """
    text = text.lower()  # Convert to lowercase
    text = ''.join(char for char in text if char.isalnum() or char.isspace())  # Remove special characters
    return text

def calculate_similarity(text1, text2):
    """
    Calculate the cosine similarity between two text documents.
    """
    # Preprocess the texts
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)

    # Vectorize the texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity[0][0]

def load_text_from_file(file_path):
    """
    Load text from a file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def main():
    # File paths for the two documents
    file1 = r"C:\Users\sachi\OneDrive\Desktop\Plagiarism\data\low\test1.txt"  # Use raw string
    file2 = r"C:\Users\sachi\OneDrive\Desktop\Plagiarism\data\low\test2.txt" # Use raw string

    # Debug: Print file paths
    print(f"Looking for file1 at: {file1}")
    print(f"Looking for file2 at: {file2}")

    # Load text from files
    try:
        text1 = load_text_from_file(file1)
        text2 = load_text_from_file(file2)
    except FileNotFoundError as e:
        print(e)
        return

    # Calculate similarity
    similarity_score = calculate_similarity(text1, text2)
    print(f"Similarity Score: {similarity_score:.4f}")

    # Interpret the result
    if similarity_score > 0.8:
        print("High similarity: Potential plagiarism detected.")
    elif similarity_score > 0.5:
        print("Moderate similarity: Possible paraphrasing.")
    else:
        print("Low similarity: No plagiarism detected.")

if __name__ == "__main__":
    main()