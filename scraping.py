# coding: UTF-8


import os
import re
import time
import urllib
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup


http_proxy = os.environ.get('HTTP_PROXY')
https_proxy = os.environ.get('HTTPs_PROXY')


proxies = {
        "http": http_proxy,
        "https": https_proxy
}


url = 'http://icvl.cs.bgu.ac.il/img/hs_pub'
save_dir = '../SCI_dataset/Download_ICVL'
os.makedirs(save_dir, exist_ok=True)


if __name__ == '__main__':
    if os.path.exists('icvl_list.html') is False:
        response = requests.get(url, proxies=proxies)
        with open('icvl_list.html', 'w') as f:
            print(response.text, file=f)
        response.encoding = response.apparent_encoding
    with open('icvl_list.html', 'r') as f:
        html_text = f.read()
    soup = BeautifulSoup(html_text, 'html.parser')
    soup_a = soup.find_all('a')
    img_list = [tag_str.text for tag_str in soup_a if tag_str.text.split('.')[-1] == 'mat']
    for name in tqdm(img_list):
        with urllib.request.urlopen(os.path.join(url, name)) as web_file:
            data = web_file.read()
            with open(os.path.join(save_dir, name), 'wb') as f:
                f.write(data)
        time.sleep(1)
