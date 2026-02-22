from setuptools import find_packages, setup
import os


def get_requirements(file_path: str) -> list:
    """Read requirements from file, skipping comments and -e installs."""
    requirements = []
    if not os.path.exists(file_path):
        return requirements
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-e"):
                requirements.append(line)
    return requirements


setup(
    name="amazon-supply-chain-intelligence",
    version="1.0.0",
    author="Supply Chain AI Team",
    author_email="supplychain-ai@example.com",
    description=(
        "AI-driven delivery delay risk prediction system using "
        "GNN + ensemble ML models for e-commerce supply chain optimization"
    ),
    long_description=open("README_WORKFLOW.md").read() if os.path.exists("README_WORKFLOW.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/supply-chain-intelligence",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "train-supply-chain=train_model:main",
        ]
    },
)
