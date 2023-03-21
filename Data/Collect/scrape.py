import requests
from bs4 import BeautifulSoup

# URL to scrape
url = "https://www.example.com"

# Keywords to search for
keywords = ["keyword1", "keyword2", "keyword3"]

# Make a request to the URL
response = requests.get(url)

# Parse the HTML content using BeautifulSoup
soup = BeautifulSoup(response.content, "html.parser")

# Find all instances of the keywords in the HTML content
found_keywords = [element.text for element in soup.find_all(text=keywords)]

# Print out the found keywords
print(found_keywords)
