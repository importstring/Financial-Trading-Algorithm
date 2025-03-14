import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging
from typing import Optional, Dict, Any, Union
import numpy as np

class ParquetHandler:
    """Handles Parquet file operations with optimized settings and error handling."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
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
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for Parquet storage."""
        for column in df.columns:
            # Convert integer-like float columns to Int64 (nullable integer)
            if df[column].dtype == 'float64':
                if df[column].notnull().all() and df[column].mod(1).eq(0).all():
                    df[column] = df[column].astype('Int64')
            
            # Convert string columns to categorical if cardinality is low
            elif df[column].dtype == 'object':
                unique_count = df[column].nunique()
                if unique_count / len(df) < 0.5:  # If less than 50% unique values
                    df[column] = df[column].astype('category')
                    
        return df
    
    def save_dataframe(self, 
                      df: pd.DataFrame, 
                      filename: str,
                      compression: str = 'snappy',
                      partition_cols: Optional[list] = None) -> Path:
        """
        Save DataFrame to Parquet format with optimized settings.
        
        Args:
            df: DataFrame to save
            filename: Name of the file (without .parquet extension)
            compression: Compression codec ('snappy', 'gzip', 'brotli', etc.)
            partition_cols: List of columns to partition by
            
        Returns:
            Path: Path where the file was saved
        """
        try:
            # Create directory if it doesn't exist
            self.base_path.mkdir(parents=True, exist_ok=True)
            
            # Optimize data types
            df = self.optimize_dtypes(df)
            
            # Prepare file path
            file_path = self.base_path / f"{filename}.parquet"
            
            # Convert to PyArrow Table for more control
            table = pa.Table.from_pandas(df)
            
            if partition_cols:
                pq.write_to_dataset(
                    table,
                    root_path=str(file_path.parent / filename),
                    partition_cols=partition_cols,
                    compression=compression
                )
            else:
                pq.write_table(
                    table,
                    file_path,
                    compression=compression,
                    use_dictionary=True,
                    compression_level=None if compression == 'snappy' else 9
                )
            
            self.logger.info(f"Successfully saved DataFrame to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame: {e}")
            raise
    
    def read_dataframe(self, 
                      filename: str,
                      columns: Optional[list] = None,
                      filters: Optional[list] = None) -> pd.DataFrame:
        """
        Read Parquet file into DataFrame with optimized settings.
        
        Args:
            filename: Name of the file (without .parquet extension)
            columns: List of columns to read
            filters: List of filters to apply during reading
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        try:
            file_path = self.base_path / f"{filename}.parquet"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {file_path}")
            
            # Read using PyArrow for better performance
            table = pq.read_table(
                file_path,
                columns=columns,
                filters=filters,
                use_threads=True
            )
            
            df = table.to_pandas()
            self.logger.info(f"Successfully read DataFrame from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to read DataFrame: {e}")
            raise
    
    def append_to_dataframe(self,
                          df: pd.DataFrame,
                          filename: str,
                          compression: str = 'snappy') -> None:
        """
        Append data to existing Parquet file.
        
        Args:
            df: DataFrame to append
            filename: Name of the file (without .parquet extension)
            compression: Compression codec
        """
        try:
            file_path = self.base_path / f"{filename}.parquet"
            
            if file_path.exists():
                # Read existing data
                existing_df = self.read_dataframe(filename)
                # Combine with new data
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                combined_df = df
            
            # Save combined data
            self.save_dataframe(combined_df, filename, compression)
            self.logger.info(f"Successfully appended data to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to append data: {e}")
            raise
    
    def get_parquet_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Get metadata information about a Parquet file.
        
        Args:
            filename: Name of the file (without .parquet extension)
            
        Returns:
            Dict containing metadata information
        """
        try:
            file_path = self.base_path / f"{filename}.parquet"
            
            if not file_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {file_path}")
            
            parquet_file = pq.ParquetFile(file_path)
            metadata = {
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': parquet_file.metadata.num_columns,
                'created_by': parquet_file.metadata.created_by,
                'schema': parquet_file.schema.to_string(),
                'size_bytes': file_path.stat().st_size,
                'last_modified': file_path.stat().st_mtime
            }
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get metadata: {e}")
            raise 