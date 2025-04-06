from typing import Any, Dict, List, Optional, Union


class DictToObject:
    """
    A utility class that converts dictionary data structures into accessible objects.
    
    This allows dictionary keys to be accessed as object attributes for more readable code.
    The conversion works recursively for nested dictionaries and lists of dictionaries.
    
    Example:
        >>> data = {"name": "Test", "details": {"id": 123, "active": True}}
        >>> obj = DictToObject(data)
        >>> obj.name
        'Test'
        >>> obj.details.id
        123
    """
    
    def __init__(self, dictionary: Dict[str, Any]) -> None:
        """
        Initialize a new object from a dictionary.
        
        Args:
            dictionary: Source dictionary to convert to an object
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                # Handle lists of dictionaries
                setattr(self, key, [DictToObject(item) for item in value])
            else:
                # Set simple values directly
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the object back to a dictionary.
        
        Returns:
            Dictionary representation of this object
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list) and value and all(isinstance(item, DictToObject) for item in value):
                result[key] = [item.to_dict() for item in value]
            else:
                result[key] = value
        return result
    
    def __str__(self) -> str:
        """String representation of the object."""
        return "\n".join(f"{key}: {value}" for key, value in self.__dict__.items())
    
    def __repr__(self) -> str:
        """Detailed string representation of the object."""
        return f"DictToObject({self.to_dict()})"
