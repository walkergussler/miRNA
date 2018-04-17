# miRNA prediction project
## Files present:
2 sets of test files - three small ones to test installation issues (smallpos.fas,smallneg.fas,smallunknown.fas), and larger ones for classification purposes (there isnt a larger unkowns file).
Additionally, there are 3 scripts. You can run this program as a regular command line scientific script with CLI.py
You could also set up a server and a client and perform operations remotely through a server if you wish.
Usage instructions are designed to print if you give malformed input or no input, so just run the program from the command line to see usage instructions. Example:

##usage

```python CLI.py```

To print accuracy numbers, try

```python CLI.py positives.fas negatives.fas```

To classify an unknown file, try

```python CLI.py positives.fas negatives.fas <your_unknown_file.fas> <method>```

where method can be 'rf' for random forest or 'km' for kmeans