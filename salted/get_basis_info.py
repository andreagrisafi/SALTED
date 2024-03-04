"""Translate density fitting basis info from FHI-aims / CP2K to SALTED density fitting basis info.
This is just an entry point for the actual implementation.
in salted/aims/get_basis_info.py and salted/cp2k/get_basis_info.py.
"""

import argparse

import inp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="run without writing to files, and print the result",
    )

    args = parser.parse_args()

    if inp.qmcode.lower() == "aims":
        from salted.aims.get_basis_info import build

        build(dryrun=args.dryrun)
    elif inp.qmcode.lower() == "cp2k":
        from salted.cp2k.get_basis_info import build

        build(dryrun=args.dryrun)
    else:
        raise ValueError(f"Unknown qmcode: {inp.qmcode}")
