import os
import re

f = open('linklist.txt')
st = f.read()
pattern = '(http://.+)'
linklst = re.findall(pattern, st)
loglst = []

for link in linklst:
    print(link)
    link = link.replace("&","\&")
    os.system('axel -n 32 '+link)
