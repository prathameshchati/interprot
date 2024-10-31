import click

from interprot.autointerp.labels2latents import labels2latents
from interprot.autointerp.pdb2labels import pdb2labels


@click.group()
def cli():
    """A tool for automatically interpreting SAEs"""
    pass


cli.add_command(pdb2labels)
cli.add_command(labels2latents)

if __name__ == "__main__":
    cli()
