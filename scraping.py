# coding: UTF-8


import os
import re
import requests
from bs4 import BeautifulSoup


http_proxy = os.environ.get('HTTP_PROXY')
https_proxy = os.environ.get('HTTPs_PROXY')

proxies = {
        "http": http_proxy,
        "https": https_proxy
}


url = 'http://icvl.cs.bgu.ac.il/img/hs_pub/'


if __name__ == '__main__':
    response = requests.get(url, proxies=proxies)
    with open('icvl_list.html', 'w') as f:
        print(response.text, file=f)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    soup_a = soup.find_all('a')
    p = re.compile(r"<[^>]*?>")
    for tag_str in soup_a:
        p.sub("", tag_str)
        print(tag_str)
