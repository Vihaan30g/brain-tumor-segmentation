class Registry:
    def __init__(self):
        self._items = {}

    def register(self, name, obj):
        self._items[name] = obj
        return obj

    def get(self, name):
        if name not in self._items:
            raise KeyError(f"Unknown registry entry: {name}")
        return self._items[name]

    def items(self):
        return self._items.items()

    def __contains__(self, name):
        return name in self._items
