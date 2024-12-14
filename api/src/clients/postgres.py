import os
from typing import List, Optional, Tuple, Any, Dict
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import connection
import logging
import requests
import pandas as pd
import json
from io import StringIO

class PostgresVectorClient:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        table_name: str = "footway_product_vector_store",
        vector_dimension: int = 1536
    ):
        self.host = host or os.getenv("POSTGRES_HOST", "postgres")
        self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
        self.database = database or os.getenv("POSTGRES_DB", "footway")
        self.user = user or os.getenv("POSTGRES_USER", "postgres")
        self.password = password or os.getenv("POSTGRES_PASSWORD", "postgres")
        self.table_name = table_name
        self.vector_dimension = vector_dimension
        self.conn: Optional[connection] = None
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """Establish connection to PostgreSQL database and ensure pgvector extension exists."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            # Create pgvector extension if it doesn't exist
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def initialize_table(self) -> None:
        """Create the vector store table if it doesn't exist."""
        if not self.conn:
            self.connect()

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            embedding vector({self.vector_dimension}),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_idx
        ON {self.table_name}
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(create_table_query)
                cur.execute(create_index_query)
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize table: {str(e)}")
            raise

    def _clean_metadata_string(self, meta_str: str) -> str:
        """
        Clean metadata string for JSON parsing.

        Args:
            meta_str: Raw metadata string to clean

        Returns:
            Cleaned metadata string ready for JSON parsing
        """
        if pd.isna(meta_str):
            return '{}'

        # Convert to string and clean
        cleaned_meta = str(meta_str)
        # Remove whitespace around colons and commas
        cleaned_meta = cleaned_meta.replace(" :", ":").replace(": ", ":")
        cleaned_meta = cleaned_meta.replace(" ,", ",").replace(", ", ",")
        # Replace single quotes with double quotes
        cleaned_meta = cleaned_meta.replace("'", '"')
        # Handle nan values
        cleaned_meta = cleaned_meta.replace(': nan,', ':null,').replace(': nan}', ':null}')
        cleaned_meta = cleaned_meta.replace(':nan,', ':null,').replace(':nan}', ':null}')

        return cleaned_meta

    def load_sample_data(self) -> None:
        """Load sample data into the vector store if the table is empty."""
        if not self.conn:
            self.connect()

        # First check if table is empty
        check_empty_query = f"""
        SELECT EXISTS (SELECT 1 FROM {self.table_name} LIMIT 1);
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(check_empty_query)
                table_has_data = cur.fetchone()[0]

                if table_has_data:
                    self.logger.info(f"Table {self.table_name} already contains data. Skipping sample data load.")
                    return

            # If we get here, the table is empty, so proceed with loading sample data
            url = "https://storage.googleapis.com/footway-plus-merchant-service-eu-prod-assets-bucket/sample_data/transformed_data_3.csv"
            response = requests.get(url)
            response.raise_for_status()

            # Read CSV into pandas DataFrame
            df = pd.read_csv(StringIO(response.text))

            # Convert metadata strings to dictionaries more safely
            metadata_dicts = []
            for meta in df['metadata']:
                try:
                    cleaned_meta = self._clean_metadata_string(meta)
                    try:
                        metadata_dict = json.loads(cleaned_meta)
                    except Exception as inner_e:
                        raise Exception(f"Failed both parsing attempts: {str(inner_e)}")
                except Exception as e:
                    self.logger.warning(f"Could not parse metadata, using empty dict instead. Error: {str(e)}\nProblematic string: {cleaned_meta[:200]}")
                    metadata_dict = {}
                metadata_dicts.append(metadata_dict)

            # Convert embedding strings to lists of floats
            embeddings = [json.loads(emb) for emb in df['embedding']]

            # Add vectors to the database using existing add_vectors method
            self.add_vectors(
                vectors=embeddings,
                contents=df['content'].tolist(),
                metadata=metadata_dicts
            )

            self.logger.info(f"Successfully loaded {len(df)} sample records into the vector store")

        except Exception as e:
            self.logger.error(f"Failed to load sample data: {str(e)}")
            raise

    def add_vectors(
        self,
        vectors: List[List[float]],
        contents: List[str],
        metadata: List[Dict[str, Any]] = None
    ) -> None:
        """
        Add multiple vectors with their associated content and metadata to the database.

        Args:
            vectors: List of embedding vectors
            contents: List of content strings associated with the vectors
            metadata: List of metadata dictionaries for each vector
        """
        if not self.conn:
            self.connect()

        if metadata is None:
            metadata = [{}] * len(vectors)

        if not (len(vectors) == len(contents) == len(metadata)):
            raise ValueError("vectors, contents, and metadata must have the same length")

        insert_query = f"""
        INSERT INTO {self.table_name} (content, metadata, embedding)
        VALUES %s
        """

        try:
            with self.conn.cursor() as cur:
                # Convert vectors to PostgreSQL array format and ensure metadata is JSON
                values = [
                    (
                        content,
                        json.dumps(meta) if isinstance(meta, dict) else meta,
                        np.array(vector).astype(float).tolist()  # Ensure vector is float array
                    )
                    for content, meta, vector in zip(contents, metadata, vectors)
                ]
                execute_values(cur, insert_query, values)
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to add vectors: {str(e)}")
            raise

    def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar vectors using cosine similarity.

        Args:
            query_vector: The vector to search for
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score (0-1) for results

        Returns:
            List of tuples containing (content, metadata, similarity_score)
        """
        if not self.conn:
            self.connect()

        search_query = f"""
        SELECT content, metadata, 1 - (embedding <=> %s::vector) as similarity
        FROM {self.table_name}
        WHERE 1 - (embedding <=> %s::vector) > %s
        ORDER BY similarity DESC
        LIMIT %s;
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(search_query, (query_vector, query_vector, similarity_threshold, limit))
                results = cur.fetchall()
                return [(row[0], row[1], row[2]) for row in results]
        except Exception as e:
            self.logger.error(f"Failed to search vectors: {str(e)}")
            raise

    def delete_vectors(self, ids: List[int]) -> None:
        """Delete vectors by their IDs."""
        if not self.conn:
            self.connect()

        delete_query = f"""
        DELETE FROM {self.table_name}
        WHERE id = ANY(%s);
        """

        try:
            with self.conn.cursor() as cur:
                cur.execute(delete_query, (ids,))
                self.conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {str(e)}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
