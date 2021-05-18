# Builtins
import datetime
from setuptools import setup

# External packages
import ruamel.yaml
from setuptools_scm import get_version

# Local imports
import os
import sys
sys.path.insert(0, ".")
from conda2pip import conda2pip

# Get necessary and optional package dependencies
required, _ = conda2pip(return_lists=True)

# Get package version for citationFile (for dev-builds this might differ from
# test-PyPI versions, which are ordered by recency)
version = get_version(root='.', relative_to=__file__, local_scheme="no-local-version")

# Update citation file
citationFile = "CITATION.cff"
yaml = ruamel.yaml.YAML()
with open(citationFile) as fl:
    ymlObj = yaml.load(fl)
ymlObj["version"] = version
ymlObj["date-released"] = datetime.datetime.now().strftime("%Y-%m-%d")
with open(citationFile, "w") as fl:
    yaml.dump(ymlObj, fl)

# Run setup (note: identical arguments supplied in setup.cfg will take precedence)
setup(
    use_scm_version={"local_scheme": "no-local-version"},
    setup_requires=['setuptools_scm'],
    install_requires=required
)
