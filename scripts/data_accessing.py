import pandas as pd
import logging

class DataAccessor:
    """
    A class to handle data access operations from a CSV file.
    
    Attributes:
        data_path (str): The path to the CSV file.
    """

    def __init__(self, data_path: str):
        """
        Initializes the DataAccessor with the path to the data file.
        
        Args:
            data_path (str): The file path for the CSV file.
        """
        self.data_path = data_path

    def read_data(self) -> pd.DataFrame:
        """
        Reads data from a CSV file into a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The data read from the CSV file.
            
        Raises:
            FileNotFoundError: If the file cannot be found.
            pd.errors.EmptyDataError: If the file is empty.
            pd.errors.ParserError: If the file is malformed.
        """
        try:
            df = pd.read_csv(self.data_path)
            logging.info("Dataset loaded successfully")
            return df
        except FileNotFoundError as e:
            print(f"Error: File not found at {self.data_path}")
            raise e
        except pd.errors.EmptyDataError as e:
            print(f"Error: The file at {self.data_path} is empty")
            raise e
        except pd.errors.ParserError as e:
            print(f"Error: Failed to parse the file at {self.data_path}")
            raise e