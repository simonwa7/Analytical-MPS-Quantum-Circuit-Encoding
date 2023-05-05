import setuptools

setuptools.setup(
    name="qcmps",
    use_scm_version=True,
    license="Apache-2.0",
    author="William A. Simon",
    author_email="william.andrew.simon@gmail.com",
    packages=setuptools.find_namespace_packages(include=["qcmps.*"], where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
)
