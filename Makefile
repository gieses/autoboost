# https://gist.github.com/prwhite/8168133
.DEFAULT_GOAL := help

.PHONY: help test test_unit test_int install_dev istall docs docs_notebooks badges notebook-reset notebooks dist pypi clean

help:                                 ## show this help
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e "s/\\$$//" | sed -e "s/##//"

test:                                 ## run tests
	pytest

install:                              ## install local millipede package
	conda env create -f environment.yml

badges:									## make the coverage badge
	coverage-badge -o .github/imgs/coverage.svg -f
	flake8 autoboost --exit-zero --htmldir=.github/imgs/ --format=html --statistics --tee --output-file .github/imgs/flake8stats.txt --exclude __init__.py
	genbadge flake8 -i .github/imgs/flake8stats.txt -o .github/imgs/flake8.svg
	genbadge tests -i .github/imgs/junit/junit.xml -o .github/imgs/tests.svg

dist: clean			      ## build the package
	python setup.py sdist bdist_wheel

pypi:				      ## prepare pypi release
	pip install twine
	python setup.py sdist
	twine upload dist/*

clean:                                ## clean up - remove docs, dist and build
	rm -r docs/build
	rm -r dist
	rm -r build
