lint:
	flake8 .

requirements:
	pip install -r requirements.txt

run:
	python -c 'import app; app.run();'
