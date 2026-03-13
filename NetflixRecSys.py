import json
import pandas as pd
import openai
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List
from textwrap import dedent

# Configuration
github_api = 'ENTER YOUR GITHUB TOKEN'
MODEL = "gpt-4o-mini"
client = OpenAI(base_url="https://models.inference.ai.azure.com",api_key= github_api,)

# Load your dataset
df = pd.read_csv('netflix_movies_detailed_up_to_2025.csv')

# --- 1. Define the Recommendation Schema ---

class MovieGenre(str, Enum):
    action = "Action"
    adventure = "Adventure"
    animation = "Animation"
    comedy = "Comedy"
    crime = "Crime"
    documentary = "Documentary"
    drama = "Drama"
    family = "Family"
    fantasy = "Fantasy"
    horror = "Horror"
    romance = "Romance"
    sci_fi = "Science Fiction"
    thriller = "Thriller"

class MovieSearchParameters(BaseModel):
    genre: Optional[MovieGenre] = None
    min_rating: Optional[float] = 0.0
    start_year: Optional[int] = 1900
    end_year: Optional[int] = 2025
    keywords: Optional[str] = None  # To search plot descriptions

# --- 2. The Search Function (The "Database" query) ---

def search_movies(params: MovieSearchParameters):
    results = df.copy()
    
    if params.genre:
        results = results[results['genres'].str.contains(params.genre.value, na=False, case=False)]
    
    # 2. Filter by Rating and Year
    results = results[
        (results['vote_average'] >= params.min_rating) &
        (results['release_year'] >= params.start_year) &
        (results['release_year'] <= params.end_year)
    ]
    
    # 3. SMART KEYWORD SEARCH (Splits the words)
    if params.keywords:
        # Turn "intense space thriller" into ['intense', 'space', 'thriller']
        keyword_list = params.keywords.lower().split()
        
        # Create a "mask" to find movies that contain ANY of those words
        mask = results['description'].str.lower().apply(
            lambda x: any(word in str(x) for word in keyword_list)
        ) | results['title'].str.lower().apply(
            lambda x: any(word in str(x) for word in keyword_list)
        )
        results = results[mask]
        
    # Sort by popularity to get the most relevant ones first
    results = results.sort_values(by='popularity', ascending=False)
    
    return results.head(3).to_dict(orient='records')

# --- 3. The Recommendation Logic ---

print("--- 🎬 Welcome to your AI Netflix Assistant ---")

user_query = input("\nWhat are you in the mood for? ") # e.g. "Scary 90s movie"
user_mood = input("Any other context? ")              # e.g. "I want something highly rated"

print("\nAI is analyzing your request...")

# This is the "Magic" step: We send the user's input to the AI
# and ask it to give us back JSON that fits our MovieSearchParameters
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system", 
            "content": "You are a movie expert. Extract search parameters from the user's request in JSON format."
        },
        {
            "role": "user", 
            "content": f"Request: {user_query}. Context: {user_mood}"
        }
    ],
    # We tell the AI exactly what keys we need
    tools=[{
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Search the movie database",
            "parameters": MovieSearchParameters.model_json_schema()
        }
    }],
    tool_choice={"type": "function", "function": {"name": "search_movies"}}
)

# --- 4. EXTRACT THE AI-GENERATED PARAMETERS ---
# Instead of you typing "Science Fiction", the AI fills this in:
tool_call = response.choices[0].message.tool_calls[0]
ai_generated_params = json.loads(tool_call.function.arguments)

print(f"AI Decision: {ai_generated_params}")

# Now we convert that JSON into our Pydantic object
final_params = MovieSearchParameters(**ai_generated_params)

# Search the CSV using the AI's choices
movies = search_movies(final_params)

# --- 5. PRINT RESULTS ---
print(f"\nTop Picks for your request:")
if not movies:
    print("No movies found. Try being less specific!")
else:
    for m in movies:
        print(f"⭐ {m['title']} ({m['release_year']}) - Rating: {m['vote_average']}")
        print(f"   {m.get('description', 'No description')[:120]}...\n")