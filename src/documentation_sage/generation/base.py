from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """
    Base interface for all LLM generators.
    """

    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        """
        Generate an answer using the query and retrieved context.
        """
        pass
