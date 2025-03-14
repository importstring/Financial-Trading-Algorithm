import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time
import logging
from typing import Dict, Any, Optional, List

class ParquetHandler:
    """Handles reading and writing Parquet format data with optimizations."""
    
    def __init__(self, base_path: Path):
        """
        Initialize ParquetHandler.
        
        Args:
            base_path: Base directory for Parquet files
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance."""
        logger = logging.getLogger("parquet_handler")
        logger.setLevel(logging.INFO)
        
        # Add file handler
        log_path = self.base_path / "parquet_handler.log"
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
        
        return logger
    
    def get_parquet_path(self, name: str) -> Path:
        """Get the full path for a Parquet file."""
        return self.base_path / f"{name}.parquet"
    
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      name: str, 
                      compression: str = 'snappy',
                      partition_cols: Optional[List[str]] = None,
                      row_group_size: int = 100000) -> str:
        """
        Save DataFrame to Parquet format with optimizations.
        
        Args:
            df: DataFrame to save
            name: Base name for the file
            compression: Compression algorithm ('snappy', 'gzip', or 'brotli')
            partition_cols: Columns to partition by
            row_group_size: Number of rows per row group
            
        Returns:
            str: Path where the data was saved
        """
        try:
            start_time = time.time()
            file_path = self.get_parquet_path(name)
            
            # Convert to PyArrow Table for optimized writing
            table = pa.Table.from_pandas(df)
            
            # Set write options
            write_options = {
                'compression': compression,
                'row_group_size': row_group_size,
                'use_dictionary': True,
                'write_statistics': True
            }
            
            if partition_cols:
                pq.write_to_dataset(
                    table,
                    root_path=str(file_path.parent / name),
                    partition_cols=partition_cols,
                    **write_options
                )
            else:
                pq.write_table(table, file_path, **write_options)
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Saved {len(df)} rows to {file_path} in {elapsed:.2f} seconds "
                f"(compression={compression})"
            )
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to Parquet: {e}")
            raise
    
    def read_dataframe(self, 
                      name: str, 
                      columns: Optional[List[str]] = None,
                      filters: Optional[List[tuple]] = None) -> pd.DataFrame:
        """
        Read DataFrame from Parquet format with optimizations.
        
        Args:
            name: Base name of the file
            columns: Specific columns to read
            filters: PyArrow filters to apply during read
            
        Returns:
            pd.DataFrame: The loaded DataFrame
        """
        try:
            start_time = time.time()
            file_path = self.get_parquet_path(name)
            
            # Check if it's a partitioned dataset
            if (file_path.parent / name).is_dir():
                dataset = pq.ParquetDataset(
                    file_path.parent / name,
                    filters=filters,
                    use_legacy_dataset=False
                )
                table = dataset.read(columns=columns)
            else:
                table = pq.read_table(
                    file_path,
                    columns=columns,
                    filters=filters
                )
            
            df = table.to_pandas()
            
            elapsed = time.time() - start_time
            self.logger.info(
                f"Read {len(df)} rows from {file_path} in {elapsed:.2f} seconds"
            )
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading DataFrame from Parquet: {e}")
            raise
    
    def get_parquet_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a Parquet file.
        
        Args:
            name: Base name of the file
            
        Returns:
            Dict containing metadata information
        """
        try:
            file_path = self.get_parquet_path(name)
            
            # Handle both partitioned and non-partitioned data
            if (file_path.parent / name).is_dir():
                dataset = pq.ParquetDataset(file_path.parent / name)
                metadata = dataset.metadata
                file_path = file_path.parent / name
            else:
                metadata = pq.read_metadata(file_path)
            
            return {
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'created_by': metadata.created_by,
                'last_modified': file_path.stat().st_mtime,
                'size_bytes': file_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"Error reading Parquet metadata: {e}")
            raise
    
    def delete_parquet(self, name: str) -> bool:
        """
        Delete a Parquet file or dataset.
        
        Args:
            name: Base name of the file
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            file_path = self.get_parquet_path(name)
            dataset_path = file_path.parent / name
            
            # Handle both partitioned and non-partitioned data
            if dataset_path.is_dir():
                import shutil
                shutil.rmtree(dataset_path)
                self.logger.info(f"Deleted partitioned dataset at {dataset_path}")
            elif file_path.exists():
                file_path.unlink()
                self.logger.info(f"Deleted file at {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting Parquet data: {e}")
            return False 