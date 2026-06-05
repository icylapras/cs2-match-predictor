#!/usr/bin/env bash
# Render runs this on every deploy: install deps, gather static files,
# apply database migrations.
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate
