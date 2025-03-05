
for file in Tutorial_1.ipynb Tutorial_1b.ipynb Tutorial_3.ipynb
do

    jupyter nbconvert --to notebook --execute $file &&
    mv ${file%.ipynb}.nbconvert.ipynb $file

done