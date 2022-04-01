import click
from repl import REPL, process_stmts


@click.command()
@click.option("--file", default=None, help="Path to a shiftql file")
@click.option("--debug", default=False, help="Print parsing results")
@click.option("--up", default=False, help="Start a server")
@click.option("--log", default=None, help="save repl log to a file")
@click.option("--times", default=1, help="Times of running the command")
def main(file, debug, up, log, times):
    for i in range(times):
        if up:
            pass
        if file is None:
            repl = REPL(debug)
            repl.run()
        else:
            with open(file, "r") as shiftql_file:
                stmts = shiftql_file.readlines()
            process_stmts(stmts, debug)


if __name__ == "__main__":
    main()
