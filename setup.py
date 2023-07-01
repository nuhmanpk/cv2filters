import pathlib
import setuptools

file = pathlib.Path(__file__).parent

README = (file / "README.md").read_text()

setuptools.setup(
    name="cv2filters",
    version="0.1.1",
    author="Nuhman Pk",
    author_email="nuhmanpk7@gmail.com",
    long_description = README,
    long_description_content_type = "text/markdown",
    description=" OpenCv Wrapper that simplifies image processing with OpenCV, making it accessible to users of all skill levels",
    license="MIT",
    url="https://github.com/nuhmanpk/cv2filters",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['cv2filters']),  
    install_requires=[
        'opencv_python'
    ],
    
    python_requires=">=3.6",
    
)