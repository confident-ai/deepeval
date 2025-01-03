from abc import ABC, abstractmethod

class BaseGuard(ABC):
    @abstractmethod
    def guard(self, *args, **kwargs) -> int:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement the 'guard' method."
        )

class BaseInputGuard(BaseGuard):
    @abstractmethod
    async def guard(self, input: str, *args, **kwargs) -> int:
        if not input:
            raise ValueError("Input must be provided and cannot be empty.")
        return super().guard(*args, **kwargs)
    
    def get_guard_type(self) -> str:
        return "InputGuard"

class BaseOutputGuard(BaseGuard):
    @abstractmethod
    async def guard(self, output: str, *args, **kwargs) -> int:
        if not output:
            raise ValueError("Output must be provided and cannot be empty.")
        return super().guard(*args, **kwargs)
    
    def get_guard_type(self) -> str:
        return "OutputGuard"
