import click

from plm_interpretability.autointerp.pdb2class import pdb2class


@click.group()
def cli():
    """A tool for automatically interpreting SAEs"""
    pass


cli.add_command(pdb2class)

if __name__ == "__main__":
    cli()
