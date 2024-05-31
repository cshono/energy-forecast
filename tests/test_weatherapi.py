import requests
import json 

url = 'https://api.weather.gov/points/39.7456,-97.0892'
response = requests.get(url)

print(response) 
print(response.json()) 