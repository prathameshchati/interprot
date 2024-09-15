import click

from plm_interpretability.autointerp.pdb2labels import pdb2labels


@click.group()
def cli():
    """A tool for automatically interpreting SAEs"""
    pass


cli.add_command(pdb2labels)

if __name__ == "__main__":
    cli()
