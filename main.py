"""main.py - command-line version. Run:  python main.py"""
from detectors import RuleBasedDetector, MLTextDetector


def show(result):
    if "error" in result:
        print("  Error:", result["error"])
        return
    print(f"  Detector  : {result['detector']}")
    print(f"  Result    : {result['label']}  ({result['confidence']}% confidence)")
    print(f"  Details   : {result['details']}")
    print(f"  Note      : {result['note']}")


def read_text():
    print("Paste your text. Type END on a new line when finished:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return " ".join(lines)


def main():
    detectors = [RuleBasedDetector(), MLTextDetector()]
    print("=== AI Text Detector ===")
    while True:
        text = read_text()
        for d in detectors:                    # same call, different behavior (polymorphism)
            print(f"\n[{d.name}]")
            try:
                show(d.analyze(text))
            except FileNotFoundError as e:
                print("  ", e)
        if input("\nCheck another text? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()