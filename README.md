# COMP345_A3_Vector_Space_Model
From the class I took about Natural Language Processing and a bit of Data Science.

Create a virtual environment
```
python3 -m venv .venv
```

Ensure you're using a version of python<3.13 since python 3.13 doesn't support numpy<2.0.0
Numpy<2.0.0 is required for gensim to work.

Install the requirements:
```
pip install -r requirements.txt
```

To convert ipynb notebooks to pdf:
```
pip install nbconvert[webpdf]
jupyter nbconvert --to webpdf --allow-chromium-download your-notebook-file.ipynb
```