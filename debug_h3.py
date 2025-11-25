import h3
import inspect

print(f"H3 Version: {h3.__version__}")

print("\nAttributes containing 'poly' or 'geo':")
for attr in dir(h3):
    if 'poly' in attr.lower() or 'geo' in attr.lower():
        print(attr)

print("\nAttributes starting with 'H3':")
for attr in dir(h3):
    if attr.startswith('H3'):
        print(attr)

