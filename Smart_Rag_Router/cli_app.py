# cli_app.py

from rag_core import process_query

while True:
    q = input("You: ")
    if q == "0":
        break

    ans, route = process_query(q)
    print(f"\n[{route}] {ans}")