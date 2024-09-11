# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging
from typing import Any, List, Optional, Tuple

from ConvAssist.utilities.logging_utility import LoggingUtility

class DatabaseError(Exception):
    """Base class for database-related errors."""
    pass

class DatabaseConnector(ABC):
    """
    Abstract base class for database interactions.
    """
    def __init__(self, logger=None):
        if logger:
            self.logger = logger
        else:
            self.logger = LoggingUtility().get_logger(__name__, log_level=logging.DEBUG)

    @abstractmethod
    def connect(self, **kwargs) -> Any:
        """
        Establish a connection to the database.
        """
        pass # pragma: no cover

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.
        """
        pass # pragma: no cover

    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> None:
        """
        Execute a query without returning any results.
        """
        pass # pragma: no cover

    @abstractmethod
    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Tuple[Any, ...]] | None:
        """
        Execute a query and return a single result.
        """
        pass # pragma: no cover

    @abstractmethod
    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Tuple[Any, ...]] | None:
        """
        Execute a query and return all results.
        """
        pass # pragma: no cover

    @abstractmethod
    def begin_transaction(self) -> None:
        """
        Begin a new database transaction.
        """
        pass # pragma: no cover

    @abstractmethod
    def commit(self) -> None:
        """
        Commit the current transaction.
        """
        pass # pragma: no cover

    @abstractmethod
    def rollback(self) -> None:
        """
        Roll back the current transaction.
        """
        pass # pragma: no cover
    
    @abstractmethod
    def create_table(self, dbname: str, tablename: str, columns: List[str]) -> None:
        """
        Create a table in the database.
        """
        pass