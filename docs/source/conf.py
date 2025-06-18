import os
import sys
import re

sys.path.insert(0, os.path.abspath('../..'))

# Get version from pyproject.toml
toml_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pyproject.toml')
try:
    # Use tomli for Python < 3.11
    try:
        import tomli
        with open(toml_path, 'rb') as f:
            pyproject = tomli.load(f)
    # Use tomllib for Python >= 3.11
    except ImportError:
        try:
            import tomllib
            with open(toml_path, 'rb') as f:
                pyproject = tomllib.load(f)
        # Fallback to regex if TOML parsing fails
        except ImportError:
            with open(toml_path, 'r') as f:
                content = f.read()
                version_match = re.search(r'version\s*=\s*["\']([^"\']*)["\'']', content)
                if version_match:
                    version = version_match.group(1)
                else:
                    version = '0.2.1'
            pyproject = {'project': {'version': version}}
    
    version = pyproject['project']['version']
except (FileNotFoundError, KeyError, Exception):
    # Fallback if reading from pyproject.toml fails
    version = '0.2.1'

# -- Project information -----------------------------------------------------
project = 'StreamPoseML'
copyright = '2025, Nate Green'
author = 'Nate Green'

# The full version, including alpha/beta/rc tags
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_multiversion',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

# -- alabaster theme options -------------------------------------------------
html_theme_options = {
    'logo': 'logo.png',  # Add your logo file to _static folder
    'github_user': 'mrilikecoding',
    'github_repo': 'StreamPoseML',
    'github_button': True,
    'github_type': 'star',
    'description': 'A toolkit for realtime video classification tasks.',
    'fixed_sidebar': True,
    'page_width': '1000px',
    'sidebar_width': '250px',
}

# -- sphinx-multiversion configuration --------------------
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'  # Include tags like v0.1.0, v1.2.3, etc.
smv_branch_whitelist = r'^(main|master|develop)$'  # Include main, master, develop branches
smv_remote_whitelist = r'^origin$'  # Only include remote 'origin'

# Add versioning template to sidebar
template_path = ['_templates']
html_sidebars = {
    '**': [
        'about.html',
        'versioning.html',  # Version selector
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

# -- autodoc configuration ---------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'
autosummary_generate = True

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

# -- Read the Docs specific settings ------------------------------------------
# This will ensure that ReadTheDocs installs your package
import os
on_rtd = os.environ.get('READTHEDOCS') == 'True'

if on_rtd:
    html_theme = 'alabaster'  # Use the same theme even on RTD
    html_theme_options = {
        'github_user': 'mrilikecoding',
        'github_repo': 'StreamPoseML',
        'github_button': True,
        'github_type': 'star',
        'description': 'A toolkit for realtime video classification tasks.',
        'fixed_sidebar': True,
    }
    
    # Manual mocking for problematic dependencies (optional)
    autodoc_mock_imports = [
        'opencv_contrib_python', 'mediapipe', 'matplotlib', 'numpy', 'pandas',
        'scikit_learn', 'scipy', 'xgboost', 'mlflow'
    ]