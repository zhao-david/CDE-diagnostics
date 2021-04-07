from setuptools import setup

import numpy as np

with open("README.md", "r") as f:
    README_TEXT = f.read()

setup(name="cde-diagnostics",
      version="0.1",
      license="MIT",
      description="Validates conditional density estimators locally and globally in feature space",
      long_description = README_TEXT,
      long_description_content_type='text/markdown; variant=GFM',
      author           = "David Zhao",
      author_email     = "davidzhao@cmu.edu",
      maintainer       = "davidzhao@cmu.edu",
      url="https://github.com/zhao-david/CDE-diagnostics",
      classifiers = ["License :: OSI Approved :: MIT License",
                     "Topic :: Scientific/Engineering :: Artificial Intelligence",
                     "Programming Language :: Python :: 2.7",
                     "Programming Language :: Python :: 3.6"],
      keywords = ["conditional density estimator", "permutation test"],
      package_dir={"": "src"},
      packages=["cde_diagnostics"],
      python_requires=">=2.7",
      install_requires=["numpy", "scipy", "sklearn"],
      setup_requires=["pytest-runner"],
      tests_require=["pytest"],
      zip_safe=False,
      include_package_data=True,
)