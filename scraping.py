# coding: UTF-8


import re
import requests
from bs4 import BeautifulSoup


proxies = {
        "http": "http://proxy.cc.yamaguchi-u.ac.jp:8080",
        "https": "http://proxy.cc.yamaguchi-u.ac.jp:8080",
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
