import re
import os
from MDRefine import __version__

def confirm():
    cont=True
    while cont:
        response=input("Confirm (yes/no)?")
        if re.match("[Yy][Ee][Ss]",response):
            cont=False
        elif re.match("[Nn][Oo]",response):
            quit()
        else:
            pass

print("Changing to master branch")

os.system("git checkout master")

print("Current version:",__version__)
new_version=re.sub("[0-9]*$","",__version__) + str(int(re.sub("^.*\.","",__version__))+1)

response=input("New version (default " + new_version + "):")

if len(response)>0:
    new_version=response

print("New version "+new_version)

confirm()

lines=[]
with open("MDRefine/_version.py") as f:
    for line in f:
        line=re.sub("^ *__version__ *=.*$",'__version__ = "' + new_version + '"',line)
        lines.append(line)

with open("MDRefine/_version.py","w") as f:
    for line in lines:
        print(line,file=f,end='')
cmd=[
    'git add MDRefine/_version.py',
    'git commit -m "Version ' + new_version + '"',
    'git tag v' + new_version,
    'git push origin master v' + new_version
]

print("Will now execute the following commands:")
for c in cmd:
    print("  " + c)

confirm()

for c in cmd:
    os.system(c)

