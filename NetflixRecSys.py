import json
import pandas as pd
import openai
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List
from textwrap import dedent

github_api = 'ENTER_GIT_API_TOKEN_HERE'
MODEL = "gpt-4o-mini"
client = OpenAI(base_url="https://models.inference.ai.azure.com",api_key= github_api,)

df = pd.read_csv('netflix_movies_detailed_up_to_2025.csv')

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
    keywords: Optional[str] = None  


def search_movies(params: MovieSearchParameters):
    results = df.copy()
    
    if params.genre:
        results = results[results['genres'].str.contains(params.genre.value, na=False, case=False)]
    
    results = results[
        (results['vote_average'] >= params.min_rating) &
        (results['release_year'] >= params.start_year) &
        (results['release_year'] <= params.end_year)
    ]
    
    if params.keywords:
       
        keyword_list = params.keywords.lower().split()
    
        mask = results['description'].str.lower().apply(
            lambda x: any(word in str(x) for word in keyword_list)
        ) | results['title'].str.lower().apply(
            lambda x: any(word in str(x) for word in keyword_list)
        )
        results = results[mask]
        
    results = results.sort_values(by='popularity', ascending=False)
    
    return results.head(3).to_dict(orient='records')

print("--- 🎬 Welcome to your AI Netflix Assistant ---")

user_query = input("\nWhat are you in the mood for? ") 
user_mood = input("Any other context? ")             

print("\nAI is analyzing your request...")

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

tool_call = response.choices[0].message.tool_calls[0]
ai_generated_params = json.loads(tool_call.function.arguments)

print(f"AI Decision: {ai_generated_params}")

final_params = MovieSearchParameters(**ai_generated_params)

movies = search_movies(final_params)

print(f"\nTop Picks for your request:")
if not movies:
    print("No movies found. Try being less specific!")
else:
    for m in movies:
        print(f"⭐ {m['title']} ({m['release_year']}) - Rating: {m['vote_average']}")
        print(f"   {m.get('description', 'No description')[:700]}...\n")