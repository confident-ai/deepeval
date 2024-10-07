from abc import ABC, abstractmethod


class AttackEnhancement(ABC):
    @abstractmethod
    def enhance(self, attack: str, *args, **kwargs) -> str:
        """Enhance the given attack synchronously."""
        pass

    async def a_enhance(self, attack: str, *args, **kwargs) -> str:
        """Enhance the given attack asynchronously."""
        return self.enhance(attack, *args, **kwargs)  # Default to sync behavior
