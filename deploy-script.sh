echo '*************************************'
date
echo 'Executing all the notebooks . . .'
jupyter nbconvert ./course/notebooks/*.ipynb --to notebook --execute --clear-output
# echo '------'
# echo 'Committing the course folder new changes . . .'
# git add -f ./course/*
# git commit -m 'Running the deploying script' && git push
echo '------'
echo 'deploying to gh-pages with mkdocs . . .'
mkdocs gh-deploy
date
