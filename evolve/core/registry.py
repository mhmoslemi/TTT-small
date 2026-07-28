"""
Generic plugin registry: example name / dotted module path -> Example instance.

Nothing problem-specific lives here; it imports example.module and calls
build(cfg). Adding an example means adding a directory, not editing this file.
"""
