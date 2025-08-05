from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

pc = Pinecone()

def context_retrieval(query, index_name, embed_model):
    index = pc.Index(index_name)
    embed_model = SentenceTransformer(embed_model)
    embedding = embed_model.encode(query).tolist()
    results = index.query(vector=embedding, top_k=5, include_metadata=True, namespace="hasan-resume")
    context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
    return context

def prompt(context, query):
    delimit="\n \n --- \n \n"
    prompt = f"""
    Answer the question based on the following context:
    {delimit}
    {context}
    {delimit}
    Question: {query}
    Answer:
    """
    return prompt