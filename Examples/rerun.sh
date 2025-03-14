
for file in Tutorial_1*ipynb Tutorial_2.ipynb
do

    jupyter nbconvert --to notebook --execute $file &&
    mv ${file%.ipynb}.nbconvert.ipynb $file

done