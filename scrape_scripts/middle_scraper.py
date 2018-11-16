from selenium import webdriver
from bs4 import BeautifulSoup
import unidecode

url = "https://en.wikipedia.org/wiki/List_of_middle-distance_runners"

driver = webdriver.Chrome()
driver.get(url)

soup = BeautifulSoup(driver.page_source, "lxml")

with open("../data/middlers.txt", "w") as f:
    div = soup.find('div', attrs={'class':'mw-parser-output'})

    for li in div.find_all('li'):
        name =li.a.get_text('title')
        f.write(unidecode.unidecode(name))
        f.write('\n')

driver.quit()
