# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../examples'))


# -- Project information -----------------------------------------------------

project = 'Olympus'
copyright = '2020, Matteo Aldeghi, Riley Hickman and Florian Häse'
author = 'Matteo Aldeghi, Riley Hickman and Florian Häse'

# The short X.Y version.
from olympus import _version
version = ".".join(_version.get_versions()['version'].split('.')[:3])
# The full version, including alpha/beta/rc tags.
release = _version.get_versions()['version']


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	'm2r',
	'nbsphinx',
	'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
	'sphinx.ext.coverage',
	'sphinx.ext.mathjax',
	'sphinx.ext.napoleon',
	'sphinx.ext.viewcode',
	'sphinx.ext.githubpages',
]

source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
	'_build', '**.ipynb_checkpoints'
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

import sphinx_rtd_theme
import msmb_theme  # this extends the sphinx theme
html_theme_path = [msmb_theme.get_html_theme_path(),
                   sphinx_rtd_theme.get_html_theme_path()]
html_theme = 'msmb_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = '_static/logo2b.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {'**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html']}

# Output file base name for HTML help builder.
htmlhelp_basename = 'olympusdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'olympus.tex', u'olympus Documentation',
     u'Matteo Aldeghi, Riley Hickman and Florian Häse', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'olympus', u'olympus Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'olympus', u'olympus Documentation',
     author, 'olympus', 'One line description of project.',
     'Miscellaneous'),
]


# concatenate and show class and __init__ docstrings
autoclass_content = 'both'


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------
autosummary_generate = True
autodoc_default_flags = ['members', 'inherited-members']

# spell checking
spelling_lang = 'en_US'
spelling_word_list_filename = 'spelling_wordlist.txt'
spelling_show_suggestions = True


def setup(app):
    app.add_stylesheet('custom.css')
    try:
        from sphinx.ext.autosummary import Autosummary
        from sphinx.ext.autosummary import get_documenter
        from docutils.parsers.rst import directives
        from sphinx.util.inspect import safe_getattr
        import re

        class AutoAutoSummary(Autosummary):

            option_spec = {
                'methods': directives.unchanged,
                'attributes': directives.unchanged
            }

            required_arguments = 1

            @staticmethod
            def get_members(obj, typ, include_public=None):
                if not include_public:
                    include_public = []
                items = []
                for name in dir(obj):
                    try:
                        documenter = get_documenter(safe_getattr(obj, name), obj)
                    except AttributeError:
                        continue
                    if documenter.objtype == typ:
                        items.append(name)
                public = [x for x in items if x in include_public or not x.startswith('_')]
                return public, items

            def run(self):
                clazz = self.arguments[0]
                try:
                    (module_name, class_name) = clazz.rsplit('.', 1)
                    m = __import__(module_name, globals(), locals(), [class_name])
                    c = getattr(m, class_name)
                    if 'methods' in self.options:
                        _, methods = self.get_members(c, 'method', ['__init__'])

                        self.content = ["~%s.%s" % (clazz, method) for method in methods if not method.startswith('_')]
                    if 'attributes' in self.options:
                        _, attribs = self.get_members(c, 'attribute')
                        self.content = ["~%s.%s" % (clazz, attrib) for attrib in attribs if not attrib.startswith('_')]
                finally:
                    return super(AutoAutoSummary, self).run()

        app.add_directive('autoautosummary', AutoAutoSummary)
    except BaseException as e:
        raise e

