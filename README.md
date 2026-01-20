# AST192
# workflow
check if installed:
git --version

clone repository (only first time):
cd <destination>
git clone git@github.com:vivikuss/AST192.git
cd AST192.git

create a new branch:
git switch main
git pull
git checkout -b <name of your new branch>

install/update the conda environment (first time or if the were changes):
conda-lock install --name ast192

before starting any work:
conda activate ast192
git switch main
git pull
git switch <name of branch>

ALWAYS CHECK BRANCH BEFORE EDITING (NEVER WORK ON main)
git branch

DON'T RUN CONDA INSTALL / PIP INSTALL INSIDE ast192, INSTEAD OPEN A PULL REQUEST / AN ISSUE ON GITHUB
this way, everyone stays in the same environment

after working on code:
git status
git add .
git commit -m “<Briefly describe update>”

push first time:
git push -u origin <name of branch> 

push every other time:
git push
