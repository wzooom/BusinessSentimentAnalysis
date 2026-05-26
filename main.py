import argparse
import importlib

STAGES = ["ingest", "preprocess", "sentiment", "lda",
          "aggregate", "brief", "report"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--company", required=True)
    args = parser.parse_args()

    module = importlib.import_module(f"src.{args.stage}")
    runner = getattr(module, f"run_{args.stage}")
    runner(args.company)


if __name__ == "__main__":
    main()
