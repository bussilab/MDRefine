
for file in Tutorial_1b.ipynb
do

    jupyter nbconvert --to notebook --execute $file &&
    mv ${file%.ipynb}.nbconvert.ipynb $file

done