# AST192
# git workflow
check if installed:
git —version

clone repository (only first time):
cd <destination>
git clone git@github.com:vivikuss/AST192.git
cd AST192.git

create a new branch:
git switch main
git pull
git checkout -b <name of your new branch>

before starting any work:
git switch main
git pull
git switch <name of branch>

ALWAYS CHECK BRANCH BEFORE EDITING (NEVER WORK ON main)
git branch

after working on code:
git status
git add .
git commit -m “<Briefly describe update>”

push first time:
git push -u origin <name of branch> 

push every other time:
git push
