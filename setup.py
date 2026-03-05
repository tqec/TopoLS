from setuptools import setup, find_packages

def read_requirements(path="requirements.txt"):
    requirements = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # skip pip-specific flags like -r or --extra-index-url
            if line.startswith(("-", "--")):
                continue
            requirements.append(line)
    return requirements


setup(
    name="topols",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=read_requirements(),
    python_requires=">=3.9",
)