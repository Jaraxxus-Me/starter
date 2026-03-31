#!/bin/bash
python -m black . --exclude "third-party|\.venv"
docformatter -i -r . --exclude venv third-party
isort . --skip-glob "third-party/*"
