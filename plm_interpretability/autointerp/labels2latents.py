import click


@click.command
def labels2latents():
    """
    Takes in a labels CSV file like this

    +----------------+----------------+
    | Sequence       | Class          |
    +----------------+----------------+
    | MVLSEGEWQL...  | 0001111110...  |
    +----------------+----------------+

    find SAE latents that tend to activate at positions with the 1 label.
    """
    pass
