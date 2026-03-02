"""
Pipeline subpackage — data acquisition through feature extraction.

Execution order
---------------
1. fetch.py      Download & quality-filter PDB structures from RCSB.
2. parse.py      Parse PDB files into typed protein/ligand atom records.
3. contacts.py   Run getcontacts, compute normalised contact-density vectors.
"""
