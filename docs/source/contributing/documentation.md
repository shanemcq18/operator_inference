# Writing Documentation

Our documentation is powered by [Jupyter Book](https://jupyterbook.org/intro.html) and GitHub pages.
Only the maintainers need to worry about GitHub pages, so this guide is about Jupyter Book and our documentation structure.

:::{admonition} Summary

- Documentation is stored as Markdown files or Jupyter notebooks in the `docs/` folder.
- Use `make docs` to compile the documentation locally.
- Do not `git add` files from `docs/_build/`.
:::

## Jupyter Books

[Jupyter Book](https://jupyterbook.org/intro.html) is "an open source project for building beautiful, publication-quality books and documents from computational material."
It allows you write technical content as Markdown files (`.md`) or Jupyter notebooks (`.ipynb`).
Use a notebook if you have code that you actually want to run (examples / tutorials); Markdown is usually sufficient and a little easier otherwise.

Install Jupyter Book with `python3 -m pip install --upgrade jupyter-book`.

## Documentation Structure

All documentation is grouped in the `docs/` folder.
The project `README.md` should be kept short and refer to the official documentation.

- `docs/_config.yml`: configuration for the Jupyter Book format.
- `docs/_toc.yml`: documentation table of contents. You will need to modify this file if you add or delete documentation pages.
- `docs/references.bib`: Bibtex citations. Cite them in the documentation with `` {cite}`referencekey` ``.
- `docs/images/`: contains all static images not generated by code in a notebook. SVGs are preferable, PDFs and PNGs are probably fine.
- `docs/source/`: contains all of the files with the actual written documentation. The file tree structure should be somewhat obvious from the structure of the page you're looking at right now and from `docs/_toc.yml`.
- `docs/source/tutorials/`: contains notebooks with full examples for specific applications.

<!-- - `docs/requirements.txt`: Software dependencies for compiling the documentation. -->

As you can see on the left of this page, `docs/source/` is organized into the following chapters.

- **Operator Inference**: general exposition about the setting and methodology.
- **Package Usage**: specifics on using the package, written in a narrative style and not as an API.
- **Tutorials and Guides**: notebooks with full examples for specific applications. These should be written in a narrative style: mathematical details are good, but only when accompanied by nontechnical summaries.
- **API Reference**: public function/class signatures and docstrings. This should be generated automatically if possible (see [jupyterbook.org/advanced/developers](https://jupyterbook.org/advanced/developers.html)).
- **Developer Guide**: instructions for developers (such as this page).

## Building Documentation Locally

From the root folder of the repository, execute `make docs` to build the documentation.
This is a shortcut for `jupyter-book build docs`; use `jupyter-book build --help` to see build options.

The documentation is processed and copied to in `docs/_build`.
Open the file `docs/_build/html/index.html` in a browser to see a preview (try Google Chrome if the file doesn't render nicely).

```{attention}
Do not `git add` the build files from `docs/_build`!
Only the `gh-pages` branch should track this folder.
The `.gitignore` should remind you of this if you accidentally try to add them.
```

## Sphinx Autodoc

Jupyter Book is essentially [an opinionated wrapper](https://jupyterbook.org/en/stable/explain/sphinx.html) around [Sphinx](https://www.sphinx-doc.org/en/master/), a program for generating Python documentation.
This project uses [Jupyter Book with Sphinx Autodoc](https://jupyterbook.org/en/stable/advanced/developers.html) to automatically generate documentation straight from code docstrings.
Because of our settings for the automatic documentation generation, please follow these guidelines.

- Class docstrings should _not_ have a "Methods" section. They may have an "Attributes" section but should not include any attributes that are formalized as properties.
- [Properties](https://docs.python.org/3/library/functions.html#property) show up automatically in the documentation, but attributes created at runtime do not.
- Use `:math:` environments to write actual math.

Note that docstrings must follow Sphinx syntax, not Jupyter Notebook syntax.
For example, use `:math:\`i^2 = -1\`` instead of `$i^2 = -1$`.

## Helpful Jupyter Book References

Some particularly useful pages from [the Jupyter Book topic guides](https://jupyterbook.org/intro.html):

- [Special content boxes (notes, tips, warnings, etc.)](https://jupyterbook.org/content/content-blocks.html)
- [List of standard admonitions](https://sphinx-book-theme.readthedocs.io/en/latest/reference/kitchen-sink/paragraph-markup.html#admonitions)
- [References and cross-references](https://jupyterbook.org/content/references.html)
- [Math and equations](https://jupyterbook.org/content/math.html)
- [Citations and bibliographies](https://jupyterbook.org/content/citations.html)
- [Images and Figures](https://jupyterbook.org/content/figures.html)