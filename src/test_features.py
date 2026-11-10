
from feature_extraction import (
    extract_features,
    FEATURE_NAMES
)

# Test 1: Standard AMP
sequence = "GLFDIVKKVVGALGSL"

features = extract_features(sequence)

assert len(features) == 422
assert len(FEATURE_NAMES) == 422

print("TEST 1: Standard AMP")
print("Feature count:", len(features))
print("PASS")


# Test 2: Short sequence
short_sequence = "ACDEF"

short_features = extract_features(short_sequence)

assert len(short_features) == 422

print("\nTEST 2: Short sequence")
print("Feature count:", len(short_features))
print("PASS")


# Test 3: Invalid amino acid
try:
    extract_features("GLFDIVKXB")
    raise AssertionError("Invalid sequence was accepted")

except ValueError:
    print("\nTEST 3: Invalid amino acid")
    print("PASS: Invalid sequence rejected")


# Test 4: Empty sequence
try:
    extract_features("")
    raise AssertionError("Empty sequence was accepted")

except ValueError:
    print("\nTEST 4: Empty sequence")
    print("PASS: Empty sequence rejected")


print("\nALL FEATURE EXTRACTION TESTS COMPLETED")
