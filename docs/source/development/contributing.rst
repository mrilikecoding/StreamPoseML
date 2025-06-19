Contributing to StreamPoseML
========================

We're excited that you're interested in contributing to StreamPoseML! This guide will help you get started as a contributor.

Ways to Contribute
----------------

There are many ways to contribute to StreamPoseML:

* **Code**: Fix bugs, implement new features, or improve performance
* **Documentation**: Help improve or translate documentation
* **Testing**: Create tests or report bugs
* **Examples**: Share your use cases or implementation examples
* **Research**: Cite us in your research or suggest new features based on research needs

Getting Started
-------------

Setting Up Your Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub

2. **Clone your fork**:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/StreamPoseML.git
      cd StreamPoseML

3. **Set up a virtual environment** (recommended):

   .. code-block:: bash

      # Create a virtual environment
      python -m venv venv
      
      # Activate it (Linux/Mac)
      source venv/bin/activate
      
      # Or on Windows
      venv\\Scripts\\activate

4. **Install in development mode**:

   .. code-block:: bash

      # Install with development dependencies
      pip install -e .[dev]
      
      # Run setup.py to set paths correctly
      python setup.py

Development Workflow
------------------

We follow the `GitHub Flow <https://guides.github.com/introduction/flow/>`_ workflow:

1. **Create a feature branch** from ``main``

   .. code-block:: bash

      git checkout main
      git pull origin main
      git checkout -b feature/your-feature-name

2. **Write your code and tests**

   Make your changes, following our coding standards (we use Black for formatting).
   Add or modify tests as needed.

3. **Run tests locally**

   .. code-block:: bash

      # Run all tests
      make test
      
      # Format your code
      make lint

4. **Commit your changes**

   Use clear, descriptive commit messages that explain what you've changed and why.

5. **Submit a pull request**

   * Push your branch to your fork
   * Create a pull request against the main repository's ``main`` branch
   * Fill out the pull request template with all relevant information

6. **Address review feedback**

   Respond to any feedback from maintainers and make necessary changes.

7. **Your PR gets merged!**

Coding Standards
--------------

* We use **Black** for code formatting
* Follow Python's `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide
* Write clear docstrings in the `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html>`_
* Include type hints where appropriate
* Update documentation for new features or changes

Testing
------

All new code should include appropriate tests:

* **Unit tests** for individual functions and classes
* **Integration tests** for component interactions
* **End-to-end tests** for complete workflows

Run tests using pytest:

.. code-block:: bash

   # Run all tests
   make test
   
   # Run specific test files
   pytest stream_pose_ml/tests/path/to/test_file.py
   
   # Run with coverage
   pytest --cov=stream_pose_ml

Documentation
-----------

Good documentation is crucial for usability:

* **Update existing docs** when changing functionality
* **Add new documentation** for new features
* **Create examples** to demonstrate usage

To build and view the documentation:

.. code-block:: bash

   # Build docs
   make docs
   
   # View in browser
   open docs/build/html/index.html

First-time Contributors
--------------------

If this is your first contribution to StreamPoseML:

1. Look for issues labeled ``good first issue`` on our `issues page <https://github.com/mrilikecoding/StreamPoseML/issues>`_
2. Introduce yourself in the issue comments and express your interest
3. A maintainer will help guide you through your first contribution

Need Help?
---------

If you have questions or need assistance:

* Ask questions in the GitHub issue for your feature
* Join our community discussions (if applicable)
* Reach out to the maintainers directly

License
------

By contributing to StreamPoseML, you agree that your contributions will be licensed under the project's `MIT License <https://github.com/mrilikecoding/StreamPoseML/blob/main/LICENSE>`_.