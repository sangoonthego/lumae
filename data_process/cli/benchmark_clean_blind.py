"""CLI tool for executing STAGE A.2.5F clean blind benchmark."""

import json
from data_process.adsqa.clean_blind_benchmark import evaluate_clean_blind_benchmark


def main() -> None:
    report = evaluate_clean_blind_benchmark()
    print("=== STAGE A.2.5F CLEAN BLIND BENCHMARK ===")
    print(f"Gate Result: {report['quality_gate']['gate_result']}")
    print(f"Comparable Samples: {report['aggregate']['comparable_sample_count']}")
    print(f"Mean tIoU: {report['aggregate']['mean_tiou']:.4f}")
    print(f"R1@0.5: {report['aggregate']['r1_0_5_fraction']} ({report['aggregate']['r1_0_5'] * 100:.2f}%)")
    print(f"R1@0.7: {report['aggregate']['r1_0_7_fraction']} ({report['aggregate']['r1_0_7'] * 100:.2f}%)")
    print(f"Semantic Correct: {report['aggregate']['semantic_correct_count']} / {report['aggregate']['comparable_sample_count']}")
    print(f"Generalization Gap: {report['dev_vs_blind']['generalization_gap']:.4f}")


if __name__ == "__main__":
    main()
