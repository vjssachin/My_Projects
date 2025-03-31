import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException
import uvicorn

# Load dataset
print("Loading hotel booking dataset...")
file_path = "/mnt/data/hotel_bookings.csv"
df = pd.read_csv(file_path)

# Data Preprocessing
print("Cleaning and preparing the data...")
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' + 
                                    df['arrival_date_month'].astype(str) + '-' +
                                    df['arrival_date_day_of_month'].astype(str), errors='coerce')
df.fillna({'children': 0, 'country': 'Unknown', 'agent': 0, 'company': 0}, inplace=True)
df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'], inplace=True)

# Analytics Functions
def revenue_trends():
    print("Analyzing revenue trends over time...")
    revenue = df.groupby(df['arrival_date'].dt.to_period('M'))['adr'].sum()
    revenue.plot(kind='line', title='Revenue Trends Over Time')
    plt.xlabel("Month")
    plt.ylabel("Total Revenue")
    plt.show()

def cancellation_rate():
    print("Calculating booking cancellation rate...")
    rate = df['is_canceled'].mean() * 100
    return f"The overall cancellation rate is {rate:.2f}% of total bookings."

def geographical_distribution():
    print("Visualizing where most bookings come from...")
    plt.figure(figsize=(10,5))
    sns.countplot(y=df['country'], order=df['country'].value_counts().index)
    plt.title("Geographical Distribution of Hotel Bookings")
    plt.xlabel("Number of Bookings")
    plt.ylabel("Country")
    plt.show()

def lead_time_distribution():
    print("Checking how far in advance bookings are made...")
    sns.histplot(df['lead_time'], bins=30, kde=True)
    plt.title("Distribution of Booking Lead Time")
    plt.xlabel("Days Before Arrival")
    plt.ylabel("Frequency")
    plt.show()

# Setting up RAG (Retrieval-Augmented Generation) using Nearest Neighbors
print("Initializing AI-powered question-answering system...")
model = SentenceTransformer('all-MiniLM-L6-v2')
knn = NearestNeighbors(n_neighbors=1, metric='cosine')

def index_data():
    print("Indexing booking records for fast search...")
    texts = df.astype(str).agg(' '.join, axis=1).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    knn.fit(embeddings)

def search_query(query):
    print(f"Processing user query: '{query}'...")
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = knn.kneighbors(query_embedding)
    return df.iloc[indices[0][0]].to_dict()

# API Development
print("Launching API service...")
app = FastAPI()

@app.post("/analytics")
def get_analytics():
    print("Generating analytics report...")
    return {
        "revenue_trends": "Revenue trends graph displayed.",
        "cancellation_rate": cancellation_rate(),
    }

@app.post("/ask")
def ask_question(query: str):
    print(f"Answering question: '{query}'")
    result = search_query(query)
    return {"response": result}

if __name__ == "__main__":
    print("System is up and running!")
    index_data()
    uvicorn.run(app, host="0.0.0.0", port=8000)