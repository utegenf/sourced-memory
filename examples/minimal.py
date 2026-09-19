from sourced_memory import SourceAwareMemory

memory = SourceAwareMemory(trusted_sources={"user"})

memory.observe("I prefer hiking on weekends.", source="user")
memory.observe("The user prefers hiking on weekends.", source="external_document")
memory.observe("The capital of France is Paris.", source="external_document")

memory.consolidate()

print("BELIEFS")
for item in memory.beliefs():
    print(f"  - {item.content} [{item.source.name}]")

print("\nCANDIDATES")
for item in memory.candidates():
    print(f"  - {item.content} [{item.source.name}]")
