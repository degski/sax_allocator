
# sax_allocator


### objective

Provide an STL compatible allocators that give a `never-moved-never-copied` guarantee. Improve locality of data. Allow flexibility.


### colony_allocator

An STL compatible allocator for use with STL 'node' containers. `colony_allocator` allocates 'nodes' from a `plf::colony` instance. The `colony_allocator` allows use of custom allocators for serving `plf::colony` (like a sandwich).

### _allocator

An STL compatible allocator for use with STL 'vector' containers. `virtual_allocator` allocates 'endless vectors' directly from virtual memory.


note:
