
# for file in Tutorial*.ipynb
# do

#     jupyter nbconvert --to notebook --execute $file &&
#     mv ${file%.ipynb}.nbconvert.ipynb $file

# done