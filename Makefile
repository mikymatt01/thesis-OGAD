.PHONY: test community-graphs-test

test:
	python -m pytest

community-graphs-test:
	python -m pytest tests/test_community_graphs.py
