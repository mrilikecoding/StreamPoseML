import os
import sys

# These modules will be mocked by autodoc_mock_imports
MOCK_MODULES = [
    'cv2', 'opencv_contrib_python', 'mediapipe', 'matplotlib', 'numpy', 'pandas', 
    'scikit_learn', 'scipy', 'xgboost', 'mlflow', 'PyWavelets', 'tslearn',
    'imbalanced_learn', 'kneed', 'tqdm', 'seaborn'
]

# Add the project root to the path so autodoc can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# Hardcode version to avoid parsing issues
version = '0.2.1'

# -- Project information -----------------------------------------------------
project = 'StreamPoseML'
copyright = '2025, Nate Green'
author = 'Nate Green'
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    # 'sphinx_multiversion',  # Disabled for now until basic docs work
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# -- alabaster theme options -------------------------------------------------
html_theme_options = {
    'github_user': 'mrilikecoding',
    'github_repo': 'StreamPoseML',
    'github_button': True,
    'github_type': 'star',
    'description': 'A toolkit for realtime video classification tasks.',
    'fixed_sidebar': True,
    'page_width': '1000px',
    'sidebar_width': '250px',
}

# -- Options for autodoc ----------------------------------------------------
autodoc_member_order = 'groupwise'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -- Read the Docs specific settings ------------------------------------------
autodoc_mock_imports = MOCK_MODULES