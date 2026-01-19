#!/usr/bin/env python3
"""
Cross-Architecture Classification Accuracy Verification

This script validates that the JPEG compression used in the microservices
architecture does not materially affect classification accuracy compared
to the raw tensor transport used in monolithic and Triton architectures.

Usage:
    # Start all three architectures first, then run:
    uv run python scripts/utils/verify_classification_accuracy.py

    # Or specify custom endpoints:
    uv run python scripts/utils/verify_classification_accuracy.py \
        --mono http://localhost:8100 \
        --micro http://localhost:8200 \
        --triton http://localhost:8300

Output:
    - Console summary with match rates and confidence deviations
    - JSON report at analysis/verification_report.json

Author: Matthew Hong
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
from tqdm import tqdm


@dataclass
class ClassificationResult:
    """Single classification result from an architecture."""
    class_id: int
    class_name: str
    confidence: float
    box_coords: tuple[float, float, float, float]  # x1, y1, x2, y2


@dataclass
class DetectionComparison:
    """Comparison of a single detection across architectures."""
    image_id: str
    detection_index: int
    mono: Optional[ClassificationResult] = None
    micro: Optional[ClassificationResult] = None
    triton: Optional[ClassificationResult] = None

    @property
    def all_match(self) -> bool:
        """Check if all available architectures agree on class_id."""
        results = [r for r in [self.mono, self.micro, self.triton] if r is not None]
        if len(results) < 2:
            return True
        return all(r.class_id == results[0].class_id for r in results)

    @property
    def mono_micro_match(self) -> Optional[bool]:
        """Check if mono and micro agree (the key comparison)."""
        if self.mono is None or self.micro is None:
            return None
        return self.mono.class_id == self.micro.class_id

    @property
    def confidence_deviation(self) -> Optional[float]:
        """Max confidence deviation between mono and micro."""
        if self.mono is None or self.micro is None:
            return None
        return abs(self.mono.confidence - self.micro.confidence)


@dataclass
class VerificationReport:
    """Aggregated verification results."""
    total_images: int = 0
    total_detections: int = 0
    mono_micro_matches: int = 0
    mono_micro_total: int = 0
    all_match_count: int = 0
    confidence_deviations: list[float] = field(default_factory=list)
    mismatches: list[dict] = field(default_factory=list)

    @property
    def mono_micro_match_rate(self) -> float:
        if self.mono_micro_total == 0:
            return 1.0
        return self.mono_micro_matches / self.mono_micro_total

    @property
    def mean_confidence_deviation(self) -> float:
        if not self.confidence_deviations:
            return 0.0
        return sum(self.confidence_deviations) / len(self.confidence_deviations)

    @property
    def max_confidence_deviation(self) -> float:
        if not self.confidence_deviations:
            return 0.0
        return max(self.confidence_deviations)

    def to_dict(self) -> dict:
        return {
            "total_images": self.total_images,
            "total_detections": self.total_detections,
            "mono_micro_comparison": {
                "matches": self.mono_micro_matches,
                "total": self.mono_micro_total,
                "match_rate_percent": round(self.mono_micro_match_rate * 100, 2),
            },
            "confidence_deviation": {
                "mean": round(self.mean_confidence_deviation, 6),
                "max": round(self.max_confidence_deviation, 6),
            },
            "mismatches": self.mismatches,
        }


def parse_response(response: dict) -> list[ClassificationResult]:
    """Parse /predict response into ClassificationResults."""
    results = []
    for det in response.get("detections", []):
        box = det["detection"]
        cls = det["classification"]
        results.append(ClassificationResult(
            class_id=cls["class_id"],
            class_name=cls["class_name"],
            confidence=cls["confidence"],
            box_coords=(box["x1"], box["y1"], box["x2"], box["y2"]),
        ))
    return results


def boxes_match(box1: tuple, box2: tuple, tolerance: float = 5.0) -> bool:
    """Check if two bounding boxes are approximately the same."""
    return all(abs(a - b) < tolerance for a, b in zip(box1, box2))


def match_detections(
    mono_results: list[ClassificationResult],
    micro_results: list[ClassificationResult],
    triton_results: list[ClassificationResult],
) -> list[DetectionComparison]:
    """Match detections across architectures by bounding box coordinates."""
    comparisons = []

    # Use mono as reference (raw tensor, no compression)
    for i, mono_det in enumerate(mono_results):
        comparison = DetectionComparison(
            image_id="",  # Set by caller
            detection_index=i,
            mono=mono_det,
        )

        # Find matching micro detection
        for micro_det in micro_results:
            if boxes_match(mono_det.box_coords, micro_det.box_coords):
                comparison.micro = micro_det
                break

        # Find matching triton detection
        for triton_det in triton_results:
            if boxes_match(mono_det.box_coords, triton_det.box_coords):
                comparison.triton = triton_det
                break

        comparisons.append(comparison)

    return comparisons


def run_verification(
    mono_url: str,
    micro_url: str,
    triton_url: str,
    images_dir: Path,
    timeout: float = 30.0,
) -> VerificationReport:
    """Run verification across all test images."""
    report = VerificationReport()

    # Get list of test images
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"No images found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} test images")
    print(f"Endpoints: mono={mono_url}, micro={micro_url}, triton={triton_url}")
    print()

    # Check connectivity
    with httpx.Client(timeout=5.0) as client:
        for name, url in [("mono", mono_url), ("micro", micro_url), ("triton", triton_url)]:
            try:
                resp = client.get(f"{url}/health")
                if resp.status_code != 200:
                    print(f"Warning: {name} health check failed: {resp.status_code}")
            except httpx.RequestError as e:
                print(f"Error: Cannot connect to {name} at {url}: {e}")
                print(f"Make sure all three architectures are running.")
                sys.exit(1)

    print("All services healthy. Starting verification...\n")

    with httpx.Client(timeout=timeout) as client:
        for image_path in tqdm(image_files, desc="Verifying"):
            image_id = image_path.stem
            report.total_images += 1

            # Read image bytes
            image_bytes = image_path.read_bytes()
            files = {"file": (image_path.name, image_bytes, "image/jpeg")}

            # Query all three architectures
            try:
                mono_resp = client.post(f"{mono_url}/predict", files=files).json()
                micro_resp = client.post(f"{micro_url}/predict", files=files).json()
                triton_resp = client.post(f"{triton_url}/predict", files=files).json()
            except httpx.RequestError as e:
                print(f"\nError processing {image_id}: {e}")
                continue

            # Parse responses
            mono_results = parse_response(mono_resp)
            micro_results = parse_response(micro_resp)
            triton_results = parse_response(triton_resp)

            # Match and compare detections
            comparisons = match_detections(mono_results, micro_results, triton_results)

            for comp in comparisons:
                comp.image_id = image_id
                report.total_detections += 1

                # Track mono vs micro comparison (the key metric)
                if comp.mono_micro_match is not None:
                    report.mono_micro_total += 1
                    if comp.mono_micro_match:
                        report.mono_micro_matches += 1
                    else:
                        report.mismatches.append({
                            "image_id": image_id,
                            "detection_index": comp.detection_index,
                            "mono": {
                                "class_id": comp.mono.class_id,
                                "class_name": comp.mono.class_name,
                                "confidence": round(comp.mono.confidence, 4),
                            },
                            "micro": {
                                "class_id": comp.micro.class_id,
                                "class_name": comp.micro.class_name,
                                "confidence": round(comp.micro.confidence, 4),
                            },
                        })

                # Track confidence deviation
                if comp.confidence_deviation is not None:
                    report.confidence_deviations.append(comp.confidence_deviation)

                # Track all-match
                if comp.all_match:
                    report.all_match_count += 1

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Verify classification accuracy across architectures"
    )
    parser.add_argument(
        "--mono", default="http://localhost:8100",
        help="Monolithic service URL (default: http://localhost:8100)"
    )
    parser.add_argument(
        "--micro", default="http://localhost:8200",
        help="Microservices gateway URL (default: http://localhost:8200)"
    )
    parser.add_argument(
        "--triton", default="http://localhost:8300",
        help="Triton gateway URL (default: http://localhost:8300)"
    )
    parser.add_argument(
        "--images", type=Path, default=Path("data/thesis_test_set"),
        help="Path to test images directory"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/verification_report.json"),
        help="Output JSON report path"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Request timeout in seconds"
    )

    args = parser.parse_args()

    # Run verification
    report = run_verification(
        mono_url=args.mono,
        micro_url=args.micro,
        triton_url=args.triton,
        images_dir=args.images,
        timeout=args.timeout,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total images processed: {report.total_images}")
    print(f"Total detections compared: {report.total_detections}")
    print()
    print("Monolithic vs Microservices (JPEG Q=95) Comparison:")
    print(f"  Top-1 class match rate: {report.mono_micro_match_rate * 100:.1f}%")
    print(f"  Mean confidence deviation: +/-{report.mean_confidence_deviation:.4f}")
    print(f"  Max confidence deviation: +/-{report.max_confidence_deviation:.4f}")
    print()

    if report.mismatches:
        print(f"Mismatches ({len(report.mismatches)}):")
        for m in report.mismatches[:5]:  # Show first 5
            print(f"  {m['image_id']}[{m['detection_index']}]: "
                  f"mono={m['mono']['class_name']} ({m['mono']['confidence']:.2f}) vs "
                  f"micro={m['micro']['class_name']} ({m['micro']['confidence']:.2f})")
        if len(report.mismatches) > 5:
            print(f"  ... and {len(report.mismatches) - 5} more")
    else:
        print("No mismatches found!")

    print()
    print("=" * 60)

    # Thesis conclusion
    if report.mono_micro_match_rate >= 0.99:
        print("CONCLUSION: JPEG compression does NOT materially affect accuracy.")
        print(f"  Match rate {report.mono_micro_match_rate*100:.1f}% >= 99% threshold")
    elif report.mono_micro_match_rate >= 0.95:
        print("CONCLUSION: Minor accuracy impact detected, document in limitations.")
        print(f"  Match rate {report.mono_micro_match_rate*100:.1f}% (95-99% range)")
    else:
        print("WARNING: Significant accuracy deviation detected!")
        print(f"  Match rate {report.mono_micro_match_rate*100:.1f}% < 95%")

    print("=" * 60)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()
