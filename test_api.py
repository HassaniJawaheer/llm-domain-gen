import requests

url = "http://localhost:8000/generate_domain"

payload = {
    "business_description": """FitFusion is a fitness studio that offers a unique blend of yoga, Pilates, and dance classes.\nTheir expert instructors design fusion 
        workouts that cater to different fitness levels, from beginners to advanced practitioners.\nTheir studios are equipped with state-of-the-art equipment and 
        providea serene atmosphere for members to relax and rejuvenate.""",
    "n_candidates": 3
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
