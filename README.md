[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/spoonsso/snell_tool/00d5408dfab776aaade97c22af609690fc1f3c96)

# snell_tool

**snell_tool** requires Python 3 and a few standard python packages. If you do not already have Python installed, I recommend the Anaconda python distribution, https://www.anaconda.com/distribution/.

Once *python* and *pip* -- the Package Installer for Python (this comes automatically with Anaconda) -- are installed, you can install **snell_tool** by executing the following command at a terminal from within the main `snell_tool` directory:

`pip install .`

Once installed, see `snell_example.ipynb` for usage. To open `snell_example.ipynb`, first run `jupyter notebook` from a terminal, and navigate to the file in jupyter's web browser application.

Users can also run the tool without installing python by clicking the `launch binder` link above.

#### The paper's figures can be generated with the notebooks in ./figures/. Note that these notebooks require additional dependencies not included in `setup.py`. The additional dependencies are listed at the beginning of each `.py` and `.ipynb` file and are included in standard python distributions, such as Anaconda.

The figure plotting notebooks run faster if the `.pickle` files found here (https://www.dropbox.com/sh/6rfhvzw4gm1jf8o/AAC5BZSCtqwMiLTTRud8mnN_a?dl=0) are downloaded and placed in the `./figures/` directory.