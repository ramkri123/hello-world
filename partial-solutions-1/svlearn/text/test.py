import text_extractor
import text_chunker
import sentence_encoder
import faiss_indexer
import os
import sys
import time
import re

INDEX_FILE_DIR = "/home/ramki/hello-world/partial-solutions-1/svlearn/text"
INDEX_FILE_NAME = "faiss_idx1"

def main():
    print("extract")
    te = text_extractor.TextExtraction.to_text("../../bgita.pdf")
    #te = text_extractor.TextExtraction.to_text("../../bhagavad-gita-in-english-source-file.pdf")
    #te = text_extractor.TextExtraction.to_text("test.txt")

    print("chunk")
    ch = text_chunker.ChunkText()
    li = ch.create_chunks(te)
    print('number of chunks', len(li))

    print("encode")
    en = sentence_encoder.SentenceEncoder()
    ve = en.encode(li)

    print("creating faiss index")
    start_time = time.time()
    faiss_index = faiss_indexer.FaissIndexer()
    #index = faiss_index.create_hnsw_index(dimension=ve.shape[1])
    index = faiss_index.create_brute_force_index(dimension=ve.shape[1])
    faiss_index.index = index
    index.add(ve)
    print(f"done creating faiss index in {round((time.time() - start_time), 2)} seconds")
    print("saving faiss index")
    start_time = time.time()
    faiss_index.save_index(f"{INDEX_FILE_DIR}/{INDEX_FILE_NAME}")
    print(f"done saving faiss index {INDEX_FILE_NAME} in {round((time.time() - start_time), 2)} seconds")

    print('***list start***')
    print(li)
    print('***list end***')

    query_embedding = en.encode(["what did Sanjaya say?"])
    index_list = faiss_index.get_neighbors(query_embedding, 4)
    output = []
    for idx, i_str in enumerate(index_list):
        i = int(i_str)
        output.append({"Rank":idx, "recodid":i, "output":li[i]})
    print('***output start***')
    print(output)
    print('***output end***')
    print(li[506])
    print(li[2551])
    print(li[1861])
    print(li[1816])

if __name__ == "__main__":
    main()
