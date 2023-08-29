tests:
	cd src && python -m unittest tests.test_to_minimal_csis -v

distr:
	python setup.py bdist_wheel
	twine upload dist/*