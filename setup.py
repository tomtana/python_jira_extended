import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="python_jira_extended",
    version="0.0.1",
    author="Thomas Fontana",
    author_email="thomas@fontana.onl",
    description="This package extendes the standard jira-python capabilities.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tomtana/python_jira_extended",
    project_urls={
        "Bug Tracker": "https://github.com/tomtana/python_jira_extended/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=["pandas", "numpy", "jira", "python-box"]
)