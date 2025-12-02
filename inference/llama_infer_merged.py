#!/usr/bin/env python3
import requests
import re

URL = "http://localhost:8000/v1/completions"

TEST_PROMPT = """You are a helpful assistant that estimates the price of a product based on its title, category, store, features, description, and details. Return only the numeric price in USD, formatted with two decimals (e.g., 19.99).

Product Title: LEZYNE Power Drive 1100i Loaded Headlight Kit Silver, One Size
Product Category: Sports & Outdoors
Store / Brand: LEZYNE
Product Features: Lezyne Power Drive 1100i Loaded Headlight Polish
Product Desver , cription: Lezyne Power Drive 1100i Loaded Headlight: Polish
Additional Details: Color SilBrand LEZYNE , Weight 0.3 Kilograms , Mounting Type Handlebar Mount , Dimensions LxWxH 6 x 5.25 x 3 inches , Auto Part Position Front , International Protection Rating IPX7 , Dimensions L x W x H 9.09 x 6.46 x 3.15 inches , Weight 0.58 Kilograms , Brand Name LEZYNE , s 1 , Manufacturer Lezyne , Part 1-LED-5A-V606_Polish/Hi Gloss , Model Year 2017 , Rank Sports & Outdoors 1667415, Bike Headlights 1296 , Available May 31, 2017

Predict the price in USD (two decimals)."""

def test_model():
    print("--- Sending RICH Context Prompt (LEZYNE Headlight) ---")
    
    payload = {
        "model": "llama-qlora",
        "prompt": TEST_PROMPT,  
        "max_tokens": 10,
        "temperature": 0.1   
    }

    try:
        response = requests.post(URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'choices' in data and len(data['choices']) > 0:
            raw_text = data['choices'][0]['text']
            print(f"Raw Model Output: '{raw_text}'")
            
            # Extract number
            numbers = re.findall(r"\d+\.?\d*", raw_text)
            if numbers:
                print(f"Parsed Price: ${numbers[0]}")
                print(f"Target Price: $99.92")
            else:
                print("No number found in output.")
        else:
            print(f"Empty Response: {data}")

    except Exception as e:
        print(f"Error: {e}")
        
def get_price_from_prompt(prompt_text):
    """
    Takes a raw prompt string, sends it to the model, 
    and returns the predicted price as a float.
    Returns 0.0 if no price is found or an error occurs.
    """
    payload = {
        "model": "llama-qlora",
        "prompt": prompt_text,
        "max_tokens": 10,
        "temperature": 0.1
    }

    try:
        response = requests.post(URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'choices' in data and len(data['choices']) > 0:
            raw_text = data['choices'][0]['text']
            
            # Extract number using regex
            numbers = re.findall(r"\d+\.?\d*", raw_text)
            if numbers:
                return float(numbers[0])
                
        return 0.0

    except Exception:
        return 0.0

if __name__ == "__main__":
    test_model()
