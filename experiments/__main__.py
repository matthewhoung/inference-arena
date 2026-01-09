"""Entry point for python -m experiments.

This allows running the experiment runner as:
    python -m experiments --help
    python -m experiments --dry-run
    python -m experiments -a monolithic -u 10 -r 1

Author: Matthew Hong
"""

from .runner import main

if __name__ == "__main__":
    main()
