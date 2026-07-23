from langchain_community.retrievers import ArxivRetriever

# Create the retrievers

retriver= ArxivRetriever(
    load_max_docs=2,    #max paper to retrieve
    load_all_available_meta=True
)

# query arxiv

doc=retriver.invoke("Large language model")

#Results

for i ,doc in enumerate(doc):
    print(f"\nResult {i+1}")
    print("Title",doc.metadata.get("Title"))
    print("Authors:",doc.metadata.get("Title"))
    print("Summary:",doc.page_content[:500])

