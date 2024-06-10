
for file in *.ipynb
do

jupyter nbconvert --to html $file

done

