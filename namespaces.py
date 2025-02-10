'''
This is just a script to list the namespaces present in our index.
'''
from config import index

# Retrieve index statistics
index_stats = index.describe_index_stats()
print(index_stats.namespaces)