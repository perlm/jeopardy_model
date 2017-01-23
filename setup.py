#!/usr/bin/python

from distutils.core import setup

setup(name='jeopardy_model',
	version='0.1.0',
	description="Jeopardy prediction model", 
	packages=['jeopardy_model'],
	install_requires=['numpy','pandas','sklearn','twitter'],
	entry_points={
		'console_scripts': [
		'jeopardy_model = jeopardy_model.__main__:main'
		]
	}
	)
