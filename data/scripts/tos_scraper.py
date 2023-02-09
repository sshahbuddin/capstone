#Turn list of companies into list of ToS json objects
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import re
from datetime import datetime
start = datetime.now()
start_time = start.strftime("%d/%m/%Y %H:%M:%S")
print("Starting ToS scraper @ "+start_time)
# Enter CSV file with a column that contains the COMPANY NAME in the 'file' variable
# Example: file = '/Users/main_user/Downloads/app_test_list.csv'
file = '_ENTER_FILENAME_WITH_FILE_PATH'
# Enter the colum that contains the company names in the 'company_name' variable 
# Example: company_column = 'app'
company_column = '_ENTER_COLUMN_WITH_COMPANY_NAME_'
def get_company_list(file,company_column):
    df = pd.read_csv(file)
    companies = df[company_column].tolist()
    return companies

def get_url(company):
    search = company + ' terms of service'
    url = 'https://www.google.com/search'
    headers = {
        'Accept' : '*/*',
        'Accept-Language': 'en-US,en;q=0.5',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
    }
    parameters = {'q': search}
    content = requests.get(url, headers = headers, params = parameters).text
    soup = BeautifulSoup(content, 'html.parser')

    search = soup.find(id = 'search')
    first_link = search.find('a')

    return first_link['href']

def get_tos_text(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text,'html.parser')
    data = soup.get_text()
    return data 

def gen_json_file():
    tos_json = []
    for i in get_company_list(file,company_column):
        url = get_url(i)
        tos_text_raw = get_tos_text(url)
        tos_text = re.sub("\s\s+"," ",(re.sub("\\n", " ",tos_text_raw)))
        tos_json.append({"Company":i,"URL":url,"ToS":tos_text})
    jsonString = json.dumps(tos_json)
    with open("tos.json","w") as tos_json:
        tos_json.write(jsonString)
##This function runs the generation of a NEW tos.json file. 
##Make sure you're running in a directory that you're ok overwriting any previous tos.json files
gen_json_file()
end = datetime.now()
end_time = end.strftime("%d/%m/%Y %H:%M:%S")
print("Ending ToS scraper @ "+end_time)
print("You should see a tos.json file in the directory you ran this script.")
