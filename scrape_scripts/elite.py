from selenium import webdriver
from bs4 import BeautifulSoup
import re
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_marathoners"

driver = webdriver.Chrome()
driver.get(url)

soup=BeautifulSoup(driver.page_source, "lxml")
div = soup.find('div', attrs={'class':'mw-parser-output'})
table = div.find_all('table')

def get_names(rows):
    names = []
    for row in rows:
        name = str(row.find('td'))
        name = name.split("<a")
        if(len(name) > 1):
            name = name[1]
            name = (name.split("</a"))[0]
            name = (name.split(">"))[1]
            names.append(name)
    return names

#Men
men = table[1]
table_body = men.find('tbody')
rows = table_body.find_all('tr')
men_names = get_names(rows)

#Women
women = table[2]
table_body = women.find('tbody')
rows = table_body.find_all('tr')
women_names = get_names(rows)

print("Best Men: ")
for man in men_names:
    print(man)

print("Best Women: ")
for women in women_names:
    print(women)

driver.quit()
