#!/bin/bash
./run_autoformat.sh
mypy .
pytest . --pylint -m pylint --pylint-rcfile=.pylintrc --ignore=third-party
pytest tests/
