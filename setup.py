import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Auto_Taste_ML",                        # 这个包的名字
    version="0.0.1",                           # 版本号
    author="Cui Zy",
    author_email="1776228595@qq.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SynchronyML/Auto_Taste_ML",
    project_urls={
        "Bug Tracker": "https://github.com/SynchronyML/Auto_Taste_ML/issues",
    },
    install_requires=[
        "numpy>=1.11.0",
        "seaborn>=0.11.2",
        "pandas>=1.3.3",
        "matplotlib>=3.4.2",
        "xgboost>=1.4.0",
        "scikit-learn>=0.24.2"
        ],                        # 依赖列表，这里需要更改
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},                         # 资源文件的名字
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8.10",
)