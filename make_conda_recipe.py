import ast
import re
from pathlib import Path

def readme():
    return Path('README.md').read_text()

def extract_variable(file_path, variable_name):
    with open(file_path, 'r') as f:
        file_content = f.read()
    module = ast.parse(file_content)
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == variable_name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Variable '{variable_name}' not found in {file_path}")

def version():
    return extract_variable('MDRefine/_version.py', '__version__')

def deps():
    return extract_variable('MDRefine/__init__.py', '_required_')

def description():
    with open('MDRefine/__init__.py', 'r') as f:
        file_content = f.read()
    module = ast.parse(file_content)
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value.split('\n')[0]
    return ""

with open('conda/meta.yaml.in') as f:
    recipe=f.read()

recipe=re.sub("__VERSION__",version(),recipe)

match=re.search("( *)(__REQUIRED__)",recipe)

requirements=""

for r in ast.literal_eval(str(deps())):
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

