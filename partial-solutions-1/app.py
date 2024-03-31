from text_extractor import *
from text_chunker import *
from sentence_encoder import *
from faiss_indexer import *
import psycopg
import faiss
import sys
import numpy

sys.path.append(os.path.abspath("/Users/shyam.govindaraj/Downloads/partial-solutions-1"))
def connectDB():
    conn = psycopg.connect(host='127.0.0.1', 
                    dbname='shyam.govindaraj', 
                    user='shyam.govindaraj', 
                    password='testing123', 
                    port=5433)
    cursor = conn.cursor()
    return conn, cursor

def create_tables():
    conn, cursor = connectDB() 
    cursor.execute("""
        DROP TABLE IF EXISTS chunks;
                """)  
    cursor.execute("""
            DROP TABLE IF EXISTS files;
                   """)  
    cursor.execute("""
        DROP EXTENSION IF EXISTS vector;
                """)  
    cursor.execute("""
        CREATE EXTENSION vector;
                """)    
    cursor.execute("""
            CREATE TABLE files (
                fileid serial PRIMARY KEY,
                name text,
                type text,
                data text)
            """)
    cursor.execute("""
            CREATE TABLE chunks (
                chunkid serial PRIMARY KEY,
                data text,
                embedding bytea,
                fileid integer)
            """)
    cursor.execute("""
            ALTER TABLE chunks ADD FOREIGN KEY ("fileid") REFERENCES "files" ("fileid")
            """)
    conn.commit()
    conn.close()

def convert_doc_to_text():
    text_list = []
    conn, cursor = connectDB() 
    print("Convert to text")
    fielname = 'test-pages.pdf'
    extract = TextExtraction.to_text("/Users/shyam.govindaraj/Downloads/pdf/test-pages.pdf")
    text_list.append(extract)

    query = """INSERT INTO files(name, type, data) VALUES(%s, %s, %s) RETURNING fileid;"""
    cursor.execute(query, (fielname, 'pdf', extract))
    conn.commit()
    conn.close()

def text_to_chunk():
    print("convert to chunks")
    conn, cursor = connectDB()
    query = """SELECT data from files WHERE fileid = 1;""" #TODO Fix the query
    cursor.execute(query)
    text_list = cursor.fetchall()
    print(text_list)

    chunked_text_list = []
    recordid_list = []
    ChText = ChunkText()
    for text_extract in text_list:
        text_chunks = ChText.create_chunks(text_extract[0])
        for text_chunk in text_chunks:
            chunked_text_list.append(text_chunk)
            query = """INSERT INTO chunks(data, fileid)
                VALUES(%s, %s) RETURNING chunkid;"""
            cursor.execute(query, (text_chunk, 1))  #TODO get ID from DB
            id = cursor.fetchone()[0]
            print(id)
            recordid_list.append(id)
    print(f"chunk size: {len(chunked_text_list)}, type chunked_text_list: {type(chunked_text_list)}")
    conn.commit()
    conn.close()

def chunk_to_embedding():
    conn, cursor = connectDB() 
    query = """SELECT data, chunkid from chunks WHERE fileid = 1;""" #TODO Fix the query
    cursor.execute(query)

    chunk_list = []
    id_list = []
    for record in cursor.fetchall():
        chunk, id = record
        chunk_list.append(chunk)
        id_list.append(id)

    sentence_encoder = SentenceEncoder()
    sentence_embedding = sentence_encoder.encode(chunk_list)
    print(f"dimension of sentence_embedding: {sentence_embedding.shape}")

    query = """UPDATE chunks SET embedding=%f WHERE chunkid = %s"""
    for idx, embedding in enumerate(sentence_embedding):
        cursor.execute(query,(embedding.tostring(), id_list[idx]))
    conn.commit()
    conn.close()
    return sentence_embedding

def embedding_to_vectorDB(sentence_embedding):
    conn, cursor = connectDB() 
    query = """SELECT embedding, chunkid from chunks WHERE fileid = 1;""" 
    cursor.execute(query)
    vector_list = []
    id_list = []
    for record in cursor.fetchall():
        embedding, id = record
        vector_list.append(embedding)
        id_list.append(id)
    
    base_index = faiss.IndexFlatL2(512) #TODO extract this from embedding
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(sentence_embedding, id_list)
    return index
                         
if __name__ == '__main__':

    print("start")
    create_tables()

    convert_doc_to_text()

    text_to_chunk()

    sentence_embedding = chunk_to_embedding()

    index = embedding_to_vectorDB(sentence_embedding) #TODO Read embedding from DB

    # Process query
    conn, cursor = connectDB() 
    sentence_encoder = SentenceEncoder()
    query_embedding = sentence_encoder.encode(["summarize the fields in ppp protocol"])
    D, I = index.search(query_embedding, 4)
    index_str_list = re.findall(r'\d+', str(I))
    query = """SELECT data from chunks WHERE chunkid = %s"""
    for idx, i_str in enumerate(index_str_list):
        i = int(i_str)
        cursor.execute(query, (i,))
        publisher_records = cursor.fetchall()
        print("output: ", i, publisher_records)
    conn.close()
