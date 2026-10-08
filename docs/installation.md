# Installation

Maskel requires Python 3.13+.

## From PyPI

```sh
pip install maskel
```

```sh
uv add maskel
```

## From GitHub (latest unreleased)

```sh
pip install git+https://github.com/bionetslab/maskel.git
```

```sh
uv add git+https://github.com/bionetslab/maskel.git
```

## From a local clone (development)

```sh
git clone https://github.com/bionetslab/maskel.git
cd maskel
uv sync                  # core only
uv sync --extra dev      # + test tools
```

To test against an unreleased checkout of the [napari-maskel](https://bionetslab.github.io/napari-maskel/) plugin instead of its PyPI release, point that repo's `uv` at this one locally — see its own installation docs.

### Shell completions

```sh
# zsh
eval "$(maskel completions zsh)"

# bash
eval "$(maskel completions bash)"

# PowerShell
maskel completions powershell | Out-String | Invoke-Expression
```

Add the appropriate line to your shell rc for persistent tab-completion.
