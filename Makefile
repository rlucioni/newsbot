deploy:
	zappa deploy prod

invoke:
	zappa invoke prod app.run

lint:
	flake8 .

prune:
	python prune.py

requirements:
	pip install -r requirements.txt

rollback:
	zappa rollback prod -n 1

run:
	python -c 'import app; app.run();'

schedule:
	zappa schedule prod

status:
	zappa status prod

logs:
	zappa tail prod --since 1d --disable-keep-open

undeploy:
	zappa undeploy prod --remove-logs

unschedule:
	zappa unschedule prod

update:
	zappa update prod
