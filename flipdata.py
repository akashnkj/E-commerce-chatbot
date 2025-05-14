import pandas as pd
import requests
import lxml
from bs4 import BeautifulSoup

url = "https://www.flipkart.com/"

r = requests.get(url)
#print(r)

soup = BeautifulSoup(r.text, "lxml")
#print(soup)

while True:
    np = soup.find("a", class_ = "_9QVEpD").get("href")
    print(np)
    cnp = "https://www.flipkart.com/"+np

    url = cnp
    r = requests.get(url)
    soup= BeautifulSoup(r.text, "lxml")


