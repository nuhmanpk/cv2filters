import pathlib
import setuptools

file = pathlib.Path(__file__).parent

README = (file / "README.md").read_text()

setuptools.setup(
    name="cv2filters",
    version="0.2.0",
    author="Nuhman Pk",
    author_email="nuhmanpk7@gmail.com",
    long_description = README,
    long_description_content_type = "text/markdown",
    description=" Unleash the power of OpenCV with CV2Filters, the ultimate image processing wrapper for all skill levels, revolutionizing computer vision with seamless exploration, manipulation, and groundbreaking research",
    license="MIT",
    url="https://github.com/nuhmanpk/cv2filters",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(include=['cv2filters']),  
    install_requires=[
        'opencv_python'
    ],
    
    python_requires=">=3.6",
    
)
