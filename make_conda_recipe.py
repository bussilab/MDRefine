import ast
import re
from MDRefine import _required_
from MDRefine import __version__

def readme():
    with open('README.md') as f:
        return f.read()

def description():
    from MDRefine import __doc__ as doc
    return doc.partition('\n')[0]

with open('conda/meta.yaml.in') as f:
    recipe=f.read()

recipe=re.sub("__VERSION__",__version__,recipe)

match=re.search("( *)(__REQUIRED__)",recipe)

requirements=""

for r in ast.literal_eval(str(_required_)):
    requirements+=match.group(1)+"- " + r+"\n"

recipe=re.sub("( *)(__REQUIRED__)\n",requirements,recipe)

recipe=re.sub("__SUMMARY__",description(),recipe)

match=re.search("( *)(__DESCRIPTION__)",recipe)

description=""

for r in readme().split("\n"):
    description+=match.group(1)+r+"\n"

recipe=re.sub("( *)(__DESCRIPTION__)",description,recipe)

with open('conda/meta.yaml',"w") as f:
    f.write(recipe)

